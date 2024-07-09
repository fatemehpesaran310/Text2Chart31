import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer
import os
import json
import pandas as pd
import wandb



parser = argparse.ArgumentParser("Supervised fine-tuning for task 2")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
)
parser.add_argument(
    "--trainset_dir", type=str, default="./prepare-data/Text2Chart-31-train.json"
)
parser.add_argument(
    "--testset_dir", type=str, default="./prepare-data/Text2Chart-31-test.json"
)
args = parser.parse_args()


output_dir = "./checkpoint/cp-task2-lora32-epoch5-llama3"

os.makedirs(output_dir, exist_ok=True)

wandb.init(name="sft-task2-lora32", project="llm-test")


def generate_prompt(dataset):
    output = []
    for csv_name, data_table, reasoning_step, description in zip(
        dataset["csv-name"],
        dataset["data-table"],
        dataset["reasoning-step"],
        dataset["description"],
    ):
        # Truncate data_table if it has more than 12 lines to avoid max token error
        data_table = (
            "\n".join(data_table.split("\n")[:12])
            if len(data_table.split("\n")) > 12
            else data_table
        )
        # Generate prompt for each data sample
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You an expert in chart generation and data visualization<|eot_id|>
<|start_header_id|>user<|end_header_id|>Given the Raw Data Table, generate the reasoning steps to determine the most suitable plot for visualizing the data, taking into account the characteristics of the data.

Raw Data Table in {csv_name} :
{data_table}

Provide the reasoning steps in the following format: 
1. Characteristics of the data and CSV file:
2. Possible plot types:
3. Most suitable plot type: 
4. Further considerations for the description: <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>{reasoning_step}<|eot_id|>

<|start_header_id|>user<|end_header_id|>Given the reasoning step above and the raw data table in {csv_name}, 
Please describe the plot you would generate to visualize this data, including: Plot type, CSV file name, raw data table, Variables assigned to each axis and Any styling, formatting, or additional elements you would include<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>{description}<|eot_id|>"""
        output.append(prompt)
    return output


model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                            load_in_8bit=False,
                                            max_memory = {0: '46GB', 1: '46GB'},
                                            device_map="cuda",
                                            torch_dtype=torch.bfloat16,
                                            use_cache=False,
                                             pretraining_tp = 1,)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
model.resize_token_embeddings(len(tokenizer))
model = get_peft_model(model, lora_config)


with open(args.trainset_dir, "r") as f:
    train_data = json.load(f)
    train_data = [data for data in train_data if data["data-type"] == "long"]
    train_data = Dataset.from_pandas(pd.DataFrame(data=train_data)).shuffle(seed=777)

with open(args.testset_dir, "r") as f:
    test_data = json.load(f)
    test_data = [data for data in test_data if data["data-type"] == "long"]
    test_data = Dataset.from_pandas(pd.DataFrame(data=test_data)).shuffle(seed=777)


optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 500
learning_rate = 5e-5
max_grad_norm = 0.3
num_train_epochs = 5.0
max_steps = -1
warmup_ratio = 0.03
evaluation_strategy = "epoch"
lr_scheduler_type = "constant"

gradient_accumulation_steps = 8
eval_accumulation_steps = 4
per_device_train_batch_size = 2
per_device_eval_batch_size = 4


training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_accumulation_steps=eval_accumulation_steps,
    optim=optim,
    evaluation_strategy=evaluation_strategy,
    save_steps=save_steps,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    max_grad_norm=max_grad_norm,
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    # group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
    
    report_to="wandb",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_total_limit=3,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=lora_config,
    formatting_func=generate_prompt,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.save_model(f"{output_dir}/final")
