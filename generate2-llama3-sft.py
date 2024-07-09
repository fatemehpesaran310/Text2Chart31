import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import argparse
import json
from peft import PeftModel
from tqdm import tqdm
import os

argparser = argparse.ArgumentParser(
    "Generate reasoning steps and descriptions for task 2"
)
argparser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
)
argparser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="./checkpoint/cp-task2-lora32-epoch5-llama3",
)
argparser.add_argument(
    "--testset_dir", type=str, default="./prepare-data/Text2Chart-31-test.json"
)
args = argparser.parse_args()


def generate_prompt_step1(dataset):
    # Truncate data_table if it has more than 50 lines to avoid token length error
    data_table = (
        "\n".join(dataset["data-table"].split("\n")[:51])
        if len(dataset["data-table"].split("\n")) > 51
        else dataset["data-table"]
    )
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You an expert in chart generation and data visualization<|eot_id|>
<|start_header_id|>user<|end_header_id|>Given the Raw Data Table, generate the reasoning steps to determine the most suitable plot for visualizing the data, taking into account the characteristics of the data.

Raw Data Table in {dataset['csv-name']} :
{data_table}

Provide the reasoning steps in the following format: 
1. Characteristics of the data and CSV file:
2. Possible plot types:
3. Most suitable plot type: 
4. Further considerations for the description: <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    return prompt


def generate_prompt_step2(dataset):
    # Truncate data_table if it has more than 7 lines to avoid token length error
    data_table = (
        "\n".join(dataset["data-table"].split("\n")[:7])
        if len(dataset["data-table"].split("\n")) > 7
        else dataset["data-table"]
    )
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You an expert in chart generation and data visualization<|eot_id|>
<|start_header_id|>user<|end_header_id|>Given the Raw Data Table, generate the reasoning steps to determine the most suitable plot for visualizing the data, taking into account the characteristics of the data.

Raw Data Table in {dataset['csv-name']} :
{data_table}

Provide the reasoning steps in the following format: 
1. Characteristics of the data and CSV file:
2. Possible plot types:
3. Most suitable plot type: 
4. Further considerations for the description: <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>{dataset['task2-generated-reasoning-step']}<|eot_id|>

<|start_header_id|>user<|end_header_id|>Given the reasoning step above and the raw data table in {dataset['csv-name']}, 
Please describe the plot you would generate to visualize this data, including: Plot type, CSV file name, raw data table, Variables assigned to each axis and Any styling, formatting, or additional elements you would include<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>"""
    return prompt


with open(args.testset_dir, "r") as f:
    data = json.load(f)
    data = [item for item in data if item["data-type"] == "long"]

model = AutoModelForCausalLM.from_pretrained(
    args.model_name, device_map="auto", load_in_8bit=False, torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
tokenizer.padding_side = "right"
model.resize_token_embeddings(len(tokenizer))

peft_model = PeftModel.from_pretrained(
    model,
    args.checkpoint_dir,
    torch_dtype="auto",
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,
)

peft_model = peft_model.merge_and_unload()
peft_model = peft_model.to("cuda")

def save_to_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


for i, d in enumerate(tqdm(data)):
    input_tokens = tokenizer(generate_prompt_step1(d), return_tensors="pt")[
        "input_ids"
    ].to("cuda")
    generated_output = peft_model.generate(
        input_ids=input_tokens,
        do_sample=True,
        top_k=10,
        temperature=1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_length=2048,
    )
    generated_output = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    d.update(
        {"task2-generated-reasoning-step": generated_output.split("assistant")[-1]}
    )

    input_tokens = tokenizer(generate_prompt_step2(d), return_tensors="pt")[
        "input_ids"
    ].to("cuda")
    generated_output = peft_model.generate(
        input_ids=input_tokens,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_length=2048,
    )
    generated_output = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    d.update({"task2-generated-description": generated_output.split("assistant")[-1]})

    save_to_json(data, fname=f"./output/{args.checkpoint_dir}.json")
