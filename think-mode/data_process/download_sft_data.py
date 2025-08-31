
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

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', default='RLAIF/OpenThoughts3-ThinkMode-SFT')
    parser.add_argument('--local_dir', default='~/data/open-thoughts-think-mode-sft')
    parser.add_argument('--think_only', action='store_true', default=False)
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    data_source = args.data_source.split('/')[-1]
    dataset = datasets.load_dataset(args.data_source)
    # filter such that response starts with <think>
    if args.think_only:
        dataset = dataset.filter(lambda x: x['response'].startswith('<think>'), num_proc=64)    
        
    split_data = dataset['train'].train_test_split(test_size=0.05)
    print(split_data)
    train_dataset = split_data['train']
    test_dataset = split_data['test']
    ability = "STEM" if args.data_source in ["RLAIF/OpenThoughts3-ThinkMode-SFT"] else "open-ended"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('prompt')
            solution = example.pop('response')            

            data = {
                "data_source": data_source,
                "prompt": question,
                "solution": solution,
                "ability": ability,                
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'solution': solution,
                    "question": question                    
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=64)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=64)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
