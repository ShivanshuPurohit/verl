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
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/big-math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--data_source", default="RLAIF/math")
    parser.add_argument("--problem_key", default="problem")
    parser.add_argument("--answer_key", default="answer")

    args = parser.parse_args()

    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = args.data_source
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if data_source in ["RLAIF/Big-Math-RL-Verified-sr0.8",
                       "RLAIF/math", "sea-snell/aime-2024",
                       "RLAIF/Big-Math-RL-Verified-sr0.3",
                       "agentica-org/DeepScaleR-Preview-Dataset"]:
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    elif data_source in ["opencompass/AIME2025"]:
        ds1 = datasets.load_dataset(data_source, "AIME2025-I", split='test')
        ds2 = datasets.load_dataset(data_source, "AIME2025-II", split='test')
        ds = datasets.concatenate_datasets([ds1, ds2])
        dataset = datasets.DatasetDict({"test": ds})

    if 'train' in dataset:
        train_dataset = dataset['train']
    else:
        train_dataset = None

    if 'test' in dataset:
        test_dataset = dataset['test']
    else:
        test_dataset = None

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop(args.problem_key)

            question = question + ' ' + instruction_following

            answer = example.pop(args.answer_key)
            data = {
                # "data_source": data_source,
                "data_source": "SynthLabsAI/Big-Math-RL-Verified",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if train_dataset:
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    if test_dataset:
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
