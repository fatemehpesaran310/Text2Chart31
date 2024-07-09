import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer
import argparse
import wandb
import os

wandb.init(name="sft-task3-lora32", project="llm-test")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

def str2bool(v):
    """Cast string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean expected")


parser = argparse.ArgumentParser("Fine-tuning script with CT2C")
parser.add_argument(
    "--dataset",
    type=str,
    help="Path to the dataset",
    default="./split-dataset.py",
)
args = parser.parse_args()


model = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_8bit=False, device_map="auto", torch_dtype=torch.bfloat16
)
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
tokenizer = AutoTokenizer.from_pretrained(model_name)

data = load_dataset(args.dataset)

data_train, data_test = data["train"], data["test"]


def generate_prompt(dataset, eos_token="</s>"):
    output = []
    for description, code in zip(dataset["description"], dataset["code"]):
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are good at describing about the given data visualization code.
Make sure when you describe a graph, mention the data points or csv file that are going to be used; otherwise, we won't be able to sketch the graph.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Your task is to generate a description of the chart based on the provided code, 
please make sure to include labels from the graph. 

Code:
“””Python
{code + ' ' if code else ''}

Please generate the corresponding description.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Description:
{description}
<|eot_id|>
"""
        output.append(prompt)
    return output


lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

print(model)

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))

model = get_peft_model(model, lora_config)


output_dir = "./checkpoint/cp-task3-lora32-epoch5-llama3-fast"

os.makedirs(output_dir, exist_ok=True)

per_device_eval_batch_size = 4
eval_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 500
learning_rate = 5e-4
max_grad_norm = 0.3
num_train_epochs = 5.0
max_steps = -1
warmup_ratio = 0.03
evaluation_strategy = "steps"
lr_scheduler_type = "constant"

training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    evaluation_strategy=evaluation_strategy,
    save_steps=save_steps,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    max_grad_norm=max_grad_norm,
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
    eval_accumulation_steps=eval_accumulation_steps,
    per_device_eval_batch_size=per_device_eval_batch_size,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=data_train,
    eval_dataset=data_test,
    peft_config=lora_config,
    formatting_func=generate_prompt,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.save_model(f"{output_dir}/final")
