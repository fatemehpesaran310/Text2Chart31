from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import time
from peft import PeftModel, LoraModel
import os


def save_to_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))

sft_model_id = "./checkpoint/cp-task1-lora16-epoch5-llama3/final/"
sft_model: LoraModel = PeftModel.from_pretrained(
    model, f"{sft_model_id}", offload_folder="lora_results/lora_7/temp"
)
sft_model = sft_model.merge_and_unload()

rl_model_id = "policy1-joint-pref-v3-2k-random-data-step63"
rl_model = PeftModel.from_pretrained(
    sft_model, f"./checkpoint/{rl_model_id}/", offload_folder="lora_results/lora_7/temp"
)

with open("./prepare-data/Text2Chart-31-test.json", "r") as file:
    test_set = json.load(file)

dataset = []
total_num = len(test_set)

# Print elapsed hour/min/sec
start = time.time()


def print_elapsed_time(start):
    elapsed = time.time() - start
    # format as 00:00:00
    print(f"=== Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} ===")


for index, item in enumerate(test_set):
    input_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are good at generating complete python code from the given chart description.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Your task is to generate a complete python code for the given description. Make sure to include all necessary libraries.

Description:
{item['description']}

Please generate the corresponding code that generates the plot that has the above description. 

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Code:
“””Python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"""
    input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")
    generated_output = rl_model.generate(
        input_ids=input_tokens,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048,
    )
    value = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    print(value)
    print(f"=== Step: ({index} / {total_num}) ===")
    print_elapsed_time(start)

    dataset.append(
        {
            "id": item["id"],
            "description": item["description"],
            "generated_output": value,
            "code": item["code"],
        }
    )
    save_to_json(dataset, f"output/{rl_model_id}.json")
