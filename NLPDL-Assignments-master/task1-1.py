import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.quantization import get_default_qconfig, float_qparams_weight_only_qconfig

from customized_gpt2 import CustomizedGPT2LMHeadModel

# using cuda:0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def naive_greedy_decoding(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    torch.cuda.empty_cache()
    
    start_time = time.time()
    out = original_model.generate(**inputs, do_sample=False, use_cache=False, max_length=MAX_NEW_LENGTH, pad_token_id=tokenizer.eos_token_id)
    
    return out, time.time() - start_time, torch.cuda.max_memory_allocated()
    
    
@torch.no_grad()
def cache_greedy_decoding(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    torch.cuda.empty_cache()
    
    start_time = time.time()
    out = original_model.generate(**inputs, do_sample=False, use_cache=True, max_length=MAX_NEW_LENGTH, pad_token_id=tokenizer.eos_token_id)
    
    return out, time.time() - start_time, torch.cuda.max_memory_allocated()

@torch.no_grad()
def quantization_greedy_decoding(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    torch.cuda.empty_cache()
    
    start_time = time.time()
    out = original_model.generate(**inputs, do_sample=False, cache_implementation="quantized", max_length=MAX_NEW_LENGTH, pad_token_id=tokenizer.eos_token_id)
    
    return out, time.time() - start_time, torch.cuda.max_memory_allocated()

@torch.no_grad()
def qandc_greedy_decoding(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    torch.cuda.empty_cache()
    
    start_time = time.time()
    out = quantization_model.generate(**inputs, do_sample=False, use_cache=True, max_length=MAX_NEW_LENGTH, pad_token_id=tokenizer.eos_token_id)
    
    return out, time.time() - start_time, torch.cuda.max_memory_allocated()

# parse command line arguments
import argparse
import pandas as pd

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="gpt2", help="model name")
    args = parser.parse_args()
 
    MAX_NEW_LENGTH = 100
    bsz = 16
    times = 0
    storages = 0

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda')
    original_model.eval()
    
    # quantization_model = original_model
    # quantization_model = quantization_model.to('cpu')
    # qconfig_8bit = get_default_qconfig('fbgemm')
    # quantization_model.config = qconfig_8bit
    # torch.quantization.prepare(quantization_model, inplace=True)
    # torch.quantization.convert(quantization_model, inplace=True)
    # quantization_model = quantization_model.to('cuda')

    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]
        
    

    for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        # naive_res, naive_time, naive_storage = naive_greedy_decoding(batch)
        
        if args.mode == "naive":
            naive_res, naive_time, naive_storage = naive_greedy_decoding(batch)
            times += naive_time
            storages = max(storages, naive_storage)
            
        elif args.mode == "cache":
            cache_res, cache_time, cache_storage = cache_greedy_decoding(batch)
            times += cache_time
            storages = max(storages, cache_storage)
            
        elif args.mode == "quantization":
            quantization_res, quantization_time, quantization_storage = quantization_greedy_decoding(batch)
            times += quantization_time
            storages = max(storages, quantization_storage)
            
        elif args.mode == "qandc":
            qandc_res, qandc_time, qandc_storage = qandc_greedy_decoding(batch)
            times += qandc_time
            storages = max(storages, qandc_storage)
            
        else:
            print("Invalid mode")
            break
        
    if args.mode == "naive":
        print("Time taken for naive greedy decoding: ", times)
        print("Max memory used for naive greedy decoding: ", storages)
        df = pd.DataFrame({'mode': [args.mode], 'time_cost': [times], 'memory_cost': [storages]})
        df.to_csv('task1-1res.csv', mode='a', header=False, index=False)
        
    elif args.mode == "cache":
        print("Time taken for cache greedy decoding: ", times)
        print("Max memory used for cache greedy decoding: ", storages)
        # add to csv task1-1-2res.csv
        df = pd.DataFrame({'mode': [args.mode], 'time_cost': [times], 'memory_cost': [storages]})
        df.to_csv('task1-1res.csv', mode='a', header=False, index=False)
        
    elif args.mode == "quantization":
        print("Time taken for quantization greedy decoding: ", times)
        print("Max memory used for quantization greedy decoding: ", storages)
        # add to csv task1-1-3res.csv
        df = pd.DataFrame({'mode': [args.mode], 'time_cost': [times], 'memory_cost': [storages]})
        df.to_csv('task1-1res.csv', mode='a', header=False, index=False)
        
    elif args.mode == "qandc":
        print("Time taken for quantization and cache greedy decoding: ", times)
        print("Max memory used for quantization and cache greedy decoding: ", storages)
        # add to csv task1-1-4res.csv
        df = pd.DataFrame({'mode': [args.mode], 'time_cost': [times], 'memory_cost': [storages]})
        df.to_csv('task1-1res.csv', mode='a', header=False, index=False)
    
    else:
        print("Invalid mode")