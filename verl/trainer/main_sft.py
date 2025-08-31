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
# TODO: this script is out of date, use https://github.com/SynthLabsAI/verl/blob/main/verl/trainer/ray_sft_trainer.py

"""
Entry point for Ray-based SFT training.
"""
from verl.trainer.ray_sft_trainer import RaySFTTrainer, Role, ResourcePoolManager
from verl.utils.fs import copy_to_local

import os
import ray
import hydra


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    run_sft(config)


def run_sft(config) -> None:
    # Set environment variables
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    if not ray.is_initialized():

        # Load in environment variables from central location
        local_env_path = copy_to_local(config.trainer.get("env", ".env"))
        from dotenv import load_dotenv
        load_dotenv(local_env_path, override=True)
        # this is for local ray cluster
        env_vars = {'TOKENIZERS_PARALLELISM': 'true', 
            'NCCL_DEBUG': 'INFO', 
            'NCCL_SOCKET_IFNAME': 'eth0', 
            'NCCL_IB_SL': '1', 
            'NCCL_IB_HCA': 'mlx5', 
            'TP_SOCKET_IFNAME': 'eth0', 
            'GLOO_SOCKET_IFNAME': 'eth0'}
        if config.trainer.nnodes == 1:
            env_vars.pop('NCCL_SOCKET_IFNAME')
            env_vars.pop('GLOO_SOCKET_IFNAME')
            env_vars.pop('TP_SOCKET_IFNAME')

        # check for HF_TOkEN or WANDB_API_KEY, if they exist in the driver process, add them to the env_vars
        if 'HF_TOKEN' in os.environ:
            env_vars['HF_TOKEN'] = os.environ['HF_TOKEN']
        if 'WANDB_API_KEY' in os.environ:
            env_vars['WANDB_API_KEY'] = os.environ['WANDB_API_KEY']
        if 'YT_TOKEN' in os.environ:
            env_vars['YT_TOKEN'] = os.environ['YT_TOKEN']
        ray.init(runtime_env={'env_vars': env_vars})
 

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)  # Make sure main_task is not scheduled on head node
def main_task(config):
    from verl.utils.fs import copy_to_local
    # Print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # Load in environment variables from central location
    local_env_path = copy_to_local(config.trainer.get("env", ".env"))
    from dotenv import load_dotenv
    load_dotenv(local_env_path, override=True)

    # Download the checkpoint from hdfs if needed
    local_path = copy_to_local(config.model.partial_pretrain)

    # Instantiate tokenizer and processor
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=config.model.trust_remote_code)

    # Define options for the ray.remote decorator
    num_gpus = 1  # Each worker needs 1 GPU
    ray_remote_options = {
        "num_gpus": num_gpus,
    }

    # Define worker class
    if config.get('strategy', 'fsdp') == 'fsdp':
        from verl.workers.sft_worker import SFTWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
        
        # Apply options to the SFTWorker class
        SFTWorker_remote = ray.remote(**ray_remote_options)(SFTWorker)
    elif config.get('strategy', 'fsdp') == 'megatron':
        # Add Megatron support if needed
        from verl.workers.megatron_sft_worker import MegatronSFTWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
        
        SFTWorker_remote = ray.remote(**ray_remote_options)(MegatronSFTWorker)
    else:
        raise NotImplementedError(f"Strategy {config.get('strategy', 'fsdp')} not supported")

    # Setup resource pool
    role_worker_mapping = {
        Role.Trainer: SFTWorker_remote  # Use the configured remote class
    }

    # Configure the resource pool based on the config - Follow the PPO pattern
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    
    mapping = {
        Role.Trainer: global_pool_id
    }

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping
    )

    # Initialize trainer
    trainer = RaySFTTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls
    )
    
    # Initialize workers and start training
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main() 
