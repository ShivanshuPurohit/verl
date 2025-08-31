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
    parser.add_argument("--max_rows", type=int, default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    # data_source = "RLAIF/open-r1-math-length-penalty-sft-10k"
    # data_source = "RLAIF/open-r1-math-sft-4k"
    data_source = "RLAIF/open-r1-math-sft-50k"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # Split into train and 1000 test examples
    dataset = dataset["train"].shuffle(seed=42).train_test_split(test_size=1000, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if args.max_rows:
        train_dataset = train_dataset.select(range(args.max_rows))
        test_dataset = test_dataset.select(range(args.max_rows))

    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')
            # system_prompt = example.pop('system')
            response = example.pop('response')
            # effort_level = example.pop('effort_level')

            # question = system_prompt + "\n\n" + question 
            # answer = example.pop('solution')
            # solution = extract_solution(answer)
            example["query"] = question
            example["response"] = response
            answer = example.pop('answer')
            data = {
                "data_source": data_source,
                "prompt": [
                {
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
                    'index': idx,
                    'response': response,
                    # 'effort_level': effort_level
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
