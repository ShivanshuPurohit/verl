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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd
from omegaconf import ListConfig
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 config):
        prompt_key = config.get('prompt_key', 'prompt')
        prompt_dict_keys = config.get('prompt_dict_keys', None)
        response_key = config.get('response_key', 'response')
        response_dict_keys = config.get('response_dict_keys', None)
        max_length = config.get('max_length', 1024)
        truncation = config.get('truncation', 'error')

        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        self.use_chat_template = config.get('use_chat_template', False)
        self.additional_mask = config.get('additional_mask', None)

        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = [] if not prompt_dict_keys else prompt_dict_keys
        self.response_dict_keys = [] if not response_dict_keys else response_dict_keys

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()
    
    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        self.prompts = self.dataframe[self.prompt_key[0] if len(self.prompt_key) == 1 else self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict            
            try:
                if isinstance(self.prompts, pd.Series):
                    self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key])
                elif isinstance(self.prompts, pd.DataFrame):
                    self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception as e:
                print("Failed to preprocess prompts, got prompts:", self.prompts)
                raise e
        self.prompts = self.prompts.tolist()

        self.responses = self.dataframe[self.response_key[0] if len(self.response_key) == 1 else self.response_key]
        for key in self.response_dict_keys:
            try:
                if isinstance(self.responses, pd.Series):
                    self.responses = self.responses.apply(lambda x: series_to_item(x)[key])
                elif isinstance(self.responses, pd.DataFrame):
                    self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print("Failed to preprocess responses, got responses:", self.responses)
                raise e
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        if self.use_chat_template:
            prompt_chat = [{'role': 'user', 'content': prompt}]        
            prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_template = "Question: {question}\n\nAnswer: "
            prompt_chat_str = prompt_template.format(question=prompt)

        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0

        # mask out part that is env feedback
        if self.additional_mask:
            for mask in self.additional_mask:
                # find all positions of mask in input_ids
                open_pattern, close_pattern = mask
                close_pattern_len = len(close_pattern + " \n\n")
                # find where the open_pattern and close_pattern are
                open_positions = [m.start() for m in re.finditer(open_pattern, response)]
                close_positions = [m.start() for m in re.finditer(close_pattern, response)]
                assert len(open_positions) == len(close_positions), f"open_positions={open_positions}, close_positions={close_positions}"
                
                for open_pos, close_pos in zip(open_positions, close_positions):
                    content_before_open = response[:open_pos]
                    # tokenize content_before_open
                    content_before_open_ids_output = tokenizer(content_before_open, return_tensors='pt', add_special_tokens=False)
                    content_before_open_ids = content_before_open_ids_output['input_ids'][0]
                    # and content wrapped by open_pattern and close_pattern
                    content_ids_output = tokenizer(response[open_pos:close_pos + close_pattern_len], return_tensors='pt', add_special_tokens=False)
                    content_ids = content_ids_output['input_ids'][0]

                    # start masking at end of `content_before_open_ids` and end at end of `content_ids`
                    mask_start = content_before_open_ids.shape[0] + prompt_length
                    mask_end = mask_start + content_ids.shape[0]
                    
                    content_str = tokenizer.decode(content_ids, skip_special_tokens=True)
                    assert "<obs>" in content_str and "</obs>" in content_str, "Masking things that are not required to be masked."
                    
                    assert (input_ids[mask_start: mask_end] == content_ids).all().item(), \
                        "Masking things that are not required to be masked."
                    
                    # this only fails when there's ' \n\n\n' at the end of content_ids, but dealt 
                    # with in the data processing step

                    loss_mask[mask_start:mask_end] = 0
                    assert loss_mask[mask_start: mask_end].sum().item() == 0, "Masking failed."

        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }
