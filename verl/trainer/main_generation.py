# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import ray
import torch
import numpy as np
import hydra
import os
import importlib

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score import _default_compute_score

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    # tokenizer.chat_template = None
    # if tokenizer.chat_template:
    #     # chat_lst = [chat.tolist() for chat in chat_lst]
    #     chat_lst = [[{"role": "user", "content": chat}] for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dispatch_dp_size = wg.world_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        if tokenizer.chat_template:
            print(batch_chat_lst)
            inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                                add_generation_prompt=True,
                                                padding=True,
                                                truncation=True,
                                                max_length=config.rollout.prompt_length,
                                                return_tensors='pt',
                                                return_dict=True,
                                                tokenize=True)
        else:
            input_ids_lst = []
            attn_mask_lst = []
            for chat in batch_chat_lst:
                input_ids = tokenizer.encode(chat, 
                                padding=True,
                                truncation=True,
                                max_length=config.rollout.prompt_length,
                                return_tensors='pt',
                                )[0]
                attn_mask = torch.tensor([1] * input_ids.shape[0])
                input_ids_lst.append(input_ids)
                attn_mask_lst.append(attn_mask)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_lst, 
                batch_first=True,
                padding_value=tokenizer.pad_token_id)
            attn_mask = torch.nn.utils.rnn.pad_sequence(
                attn_mask_lst, 
                batch_first=True,
                padding_value=tokenizer.pad_token_id)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attn_mask
            }
                
                            
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dispatch_dp_size != 0:
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # START TO GENERATE FOR n_samples TIMES
        for i in range(config.data.n_samples):
            output = wg.generate_sequences(data)
            # remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                 skip_special_tokens=False)

            # remove the padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst[i].extend(output_text_unpad)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    print(output_lst)
    print(len(dataset), len(output_lst))
    # add to the data frame
    dataset[f'responses'] = output_lst

    # compute scores if enabled
    if config.get("compute_scores", False):
        print("Computing scores for responses...")
        data_source = dataset['data_source'].iloc[0] if 'data_source' in dataset.columns else None
        scores = []
        
        for i, row in dataset.iterrows():
            ground_truth = row['reward_model']['ground_truth'] if 'reward_model' in row and 'ground_truth' in row['reward_model'] else None
            extra_info = row.get('extra_info', None)
            
            if ground_truth is not None:
                # Compute score for each response (n_samples)
                row_scores = []
                for response in row['responses']:
                    score = _default_compute_score(data_source, response, ground_truth, extra_info)
                    row_scores.append(score)
                scores.append(row_scores)
            else:
                scores.append([0.0] * len(row['responses']))
                
        dataset['scores'] = scores
        
        # Calculate average score
        avg_scores = []
        for response_scores in scores:
            avg_scores.append(sum(response_scores) / len(response_scores) if response_scores else 0)
        
        print(f"Average score: {sum(avg_scores) / len(avg_scores):.4f}")

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)

    return output_text


if __name__ == '__main__':
    main()