import argparse
import numpy as np

from tqdm import tqdm
from langdetect import detect
from datasets import load_dataset, Dataset



def is_english(text):
    """
    Check if text is in English using language detection
    """
    try:
        text_sample = text.replace('<think>', '').replace('</think>', '')
        if len(text_sample.strip()) < 10:  # Skip very short texts
            return False
        detected_lang = detect(text_sample)
        return detected_lang == 'en'
    except Exception as e:                
        return False


def preprocess_data(dataset_name, split, output_path):    
    if dataset_name == "open-thoughts/OpenThoughts3-1.2M":
        def extract_messages(example):
            conversations = example['conversations']
            
            for conv in conversations:
                if conv['from'] == 'human':
                    human_message = conv['value']
                elif conv['from'] == 'gpt':
                    model_message = conv['value']
            
            example['human_message'] = human_message
            example['model_message'] = model_message
            return example
        
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        ds = ds.map(extract_messages, num_proc=128)
        ds = ds.filter(lambda x: x['model_message'].count("<think>") > 0, num_proc=128)
        ds = ds.filter(lambda x: x['model_message'].count("<think>") == x['model_message'].count("</think>"), num_proc=128)
        
        def extract_post_thinking_message(example):
            """
            Extract the message after the thinking tags
            """
            model_message = example['model_message']
            try:
                example['model_message_no_thinking'] = model_message.split('</think>')[1].lstrip()
            except Exception as e:                
                example['model_message_no_thinking'] = ""
            return example
        
        ds = ds.map(extract_post_thinking_message, num_proc=128)
        ds = ds.filter(lambda x: x['model_message_no_thinking'], num_proc=128)        
        
        def filter_english_only(example):
            """
            Filter function to keep only English messages
            """
            human_is_english = is_english(example['human_message'])
            model_is_english = is_english(example['model_message_no_thinking'])
            return human_is_english and model_is_english
            
        ds = ds.filter(filter_english_only, num_proc=128)
        
        def finalize_dataset(ds):
            new_data = {
                    'difficulty': [],
                    'source': [],
                    'domain': [],
                    'prompt': [],
                    'response': []
                }
                
            for example in ds:
                # First row: with thinking
                new_data['difficulty'].append(example['difficulty'])
                new_data['source'].append(example['source'])
                new_data['domain'].append(example['domain'])
                new_data['prompt'].append(example['human_message'])
                new_data['response'].append(example['model_message'])
                
                # Second row: without thinking
                new_data['difficulty'].append(example['difficulty'])
                new_data['source'].append(example['source'])
                new_data['domain'].append(example['domain'])
                new_data['prompt'].append(example['human_message'])
                new_data['response'].append(example['model_message_no_thinking'])
            
            return Dataset.from_dict(new_data)
        
        print("Old dataset size: ", len(ds))
        new_ds = finalize_dataset(ds)
        print(f"New dataset size (doubled): {len(new_ds)}")
        new_ds.push_to_hub(output_path)
        
    elif dataset_name == "pharaouk/CoT-Collection":
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)                
        breakpoint()
    elif dataset_name == "ServiceNow-AI/R1-Distill-SFT":
        ds = load_dataset(dataset_name, 'v1',split=split, trust_remote_code=True)
        ds = ds.filter(lambda x: x['source_dataset'] == "allenai/tulu-3-sft-mixture", num_proc=128)

        def filter_english_only(example):
            """
            Filter function to keep only English messages
            """
            human_is_english = is_english(example['reannotated_messages'][0]['content'])
            model_is_english = is_english(example['reannotated_messages'][1]['content'])
            return human_is_english and model_is_english

        ds = ds.filter(filter_english_only, num_proc=128)

        def filter_empty_thinking(example):
            """
            Filter function to keep only rows with thinking
            """
            assistant_message = example['reannotated_messages'][1]['content']            
            if assistant_message.count("<think>") and (assistant_message.count("<think>") == assistant_message.count("</think>")):
                try:
                    thinking_content = assistant_message.split("<think>")[1].split("</think>")[0].strip()
                except Exception as e:
                    thinking_content = ""

                return thinking_content != ""
            else:
                return False

        ds = ds.filter(filter_empty_thinking, num_proc=128)
                
        def finalize_dataset(ds):
            new_data = {
                    'source': [],
                    'prompt': [],
                    'response': []
                }
                        
            for example in tqdm(ds):
                prompt = example['reannotated_messages'][0]['content']
                response_no_thinking = example['reannotated_messages'][1]['content'].split("</think>")[1].lstrip()
                response = example['reannotated_messages'][1]['content']
                
                # with thinking
                new_data['source'].append(example['source_dataset'])
                new_data['prompt'].append(prompt)
                new_data['response'].append(response)
                
                # without thinking
                new_data['source'].append(example['source_dataset'])
                new_data['prompt'].append(prompt)
                new_data['response'].append(response_no_thinking)
            
            return Dataset.from_dict(new_data)
        
        print("Old dataset size: ", len(ds))
        new_ds = finalize_dataset(ds)
        print(f"New dataset size (doubled): {len(new_ds)}")
        new_ds.push_to_hub(output_path)

    elif dataset_name == "isaiahbjork/chain-of-thought":
        breakpoint()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ServiceNow-AI/R1-Distill-SFT", 
                        choices=["open-thoughts/OpenThoughts3-1.2M", "pharaouk/CoT-Collection", 
                        "ServiceNow-AI/R1-Distill-SFT", "isaiahbjork/chain-of-thought"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="RLAIF/ServiceNow-ThinkMode-SFT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_data(args.dataset_name, args.split, args.output_path)