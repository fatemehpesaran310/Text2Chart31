import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy

tqdm.pandas()
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from trl import AutoModelForCausalLMWithValueHead
from trl import PPOTrainer, PPOConfig
from evaluate import load
from peft import PeftModel, LoraModel
from typing import List
import json

import regex as re
import subprocess

wandb.init(name="joint-p1-pref-p3-align", project="llm-test")

ds = load_dataset("./split-dataset.py")


def prompt_task1(item):
    item[
        "description"
    ] = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
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
    return item


def prompt_task2(cleaned_codes):
    new_prompt = []
    for code in cleaned_codes:
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
"""

        new_prompt.append(prompt)

    return new_prompt


ds["train"] = ds["train"].map(prompt_task1, batched=False)


def extract_desc(original_prompts):
    start_phrase = "Description:"
    end_phrase = "\nPlease generate the corresponding"

    cleaned_descriptions = []
    for string in original_prompts:
        start_index = string.find(start_phrase)
        end_index = string.find(end_phrase)
        if start_index != -1 and end_index != -1:
            desired_string = string[start_index + len(start_phrase) : end_index].strip()
            cleaned_descriptions.append(desired_string)

    return cleaned_descriptions


def extract_description_from_policy3(text_list):
    descriptions = []

    for text in text_list:
        # Define the pattern to match "Description:" followed by any characters until the end of the string
        pattern = r"Description:(.*)"
        # Use re.search to find the first occurrence of the pattern in the text
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # Extract the matched substring
            description = match.group(1).strip()
            description = re.sub(r"<.*?s>", "", description)
            descriptions.append(description)
        else:
            descriptions.append("")
    # Remove "\n<|eot_id|>" from the description if present
    descriptions = [
        description.replace("\n<|eot_id|>", "") for description in descriptions
    ]

    return descriptions


### Extract the code:
def extract_code(text):
    text = text.replace("python", "")
    text = text.replace("Python", "")
    # Define a regex pattern that matches code enclosed in ```
    # Matches with or without the python markdown
    pattern1 = r"```?\n(.*?)\n```"
    pattern2 = r'"""?\n(.*?)"""'
    pattern3 = r"“””?\n(.*?)“””"
    # Find matches using the regex pattern
    matches = re.findall(pattern1, text, re.DOTALL)

    if not matches:
        matches = re.findall(pattern2, text, re.DOTALL)
    if not matches:
        matches = re.findall(pattern3, text, re.DOTALL)
    if not matches:
        return text

    # If <\/s> is present, remove it
    matches = [match.replace("<\/s>", "") for match in matches]

    # Only keep matches that span more than one line
    multiline_matches = [match for match in matches if "\n" in match]
    return "\n".join(multiline_matches)


def clean(text):
    text = text.replace("plt.\n", "\n")
    text = text.replace("pl\n", "\n")
    # If the text has any token that has <[something]s>, remove it
    text = re.sub(r"<.*?s>", "", text)
    for target in ["Python", "python", "```", "“””", '"""', "'''", "‘", "’"]:
        text = text.replace(target, "")
    return text


def extract_output(data):
    modified_data = []
    data_code = []
    for itm in data:
        output = extract_code(itm)
        data_code.append(output)

    for item in data:
        output = extract_code(item)
        import_numpy_index = output.find("import numpy as np")
        if import_numpy_index == -1:
            output = "import numpy as np\n" + output

        import_pandas_index = output.find("import pandas as pd")
        if import_pandas_index == -1:
            output = "import pandas as pd\n" + output

        matplotlib_index = output.find("import matplotlib\n")
        matplotlib_plt_index = output.find("import matplotlib.pyplot as plt")
        if matplotlib_plt_index == -1 and matplotlib_index == -1:
            output = "import matplotlib.pyplot as plt\n" + output

        import_index = output.find("import")
        show_index = output.find("plt.show()")

        if import_index != -1 and show_index != -1:
            modified_output = output[import_index : show_index + len("plt.show()")]
        elif import_index != -1:
            modified_output = output[import_index:] + "\nplt.show()"
        else:
            print(f"output: {output}")
            print(f"import_index: {import_index}, show_index: {show_index}")
        modified_output = clean(modified_output)
        # item['cleaned-output'] = modified_output
        modified_data.append(modified_output)

    return modified_data


def save_description_dicts(
    original_descriptions, regenerated_descriptions, rewards, filename
):
    description_dicts = []
    for original_desc, regenerated_desc, reward in zip(
        original_descriptions, regenerated_descriptions, rewards
    ):
        description_dict = {
            "original-desc": original_desc,
            "regenerated-desc": regenerated_desc,
            "reward": reward,
        }
        description_dicts.append(description_dict)
    # save the json file
    with open(filename, "w") as file:
        json.dump(description_dicts, file)

    return description_dicts


def save_code(code: str):
    code = code.replace("plt.show()", "# plt.show()")
    with open("generated-policy1.py", "w") as f:
        f.write(code)
        f.write("\nplt.savefig('generated-figure-policy1.png')")
        f.write("\n")


# Function to Check the error:
def check_error(cleaned_codes, id_list):
    reward = []
    for code, id in zip(cleaned_codes, id_list):
        if os.path.isdir(f"./files/{id}"):
            os.chdir(f"./files/{id}")
            save_code(code)

            try:
                # Run the code in a separate process
                subprocess.run(["python", "generated-policy1.py"], check=True)
                # Change back to the original directory
                os.chdir("../..")
                reward.append(0.1)

            except subprocess.CalledProcessError as e:
                print(
                    "\033[93m"
                    + f"Error in script execution for {id}: {str(e)}"
                    + "\033[37m"
                )
                reward.append(-0.1)
                # Ensure to change back to the original directory in case of an error
                os.chdir("../..")
            except Exception as e:
                print(f"Error in case {id}: {str(e)}")
                reward.append(-0.1)
                os.chdir("../..")
        else:
            reward.append(0)
    return reward


# Reward function: Bert Score:
def align_reward_func(predictions: List[str], references: List[str]):
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
    )
    return [(val - 0.65) * 10 for val in results["recall"]]


# # Reward function: Preference score:
def pref_reward_func_1(
    model: AutoModelForSequenceClassification,
    predictions: List[str],
    original_descriptions: List[str],
):
    prompts = [f"Code:\n{pred}\n" for pred in predictions]

    rewards = []
    for prompt in prompts:
        input_ids = tokenizer_reward1(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        logits = model(input_ids=input_ids).logits
        rewards.append(logits.tolist()[0][0])
    return rewards


# Load reward model for policy 1:
reward_model_path_policy1 = (
    "./checkpoint/cp-task1-llama3-opt-reward-model-v3/final"
)
reward_model_1 = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path_policy1,
    num_labels=1,
    device_map="cuda",
)
tokenizer_reward1 = AutoTokenizer.from_pretrained(reward_model_path_policy1)


# Load model LLM_1:
model_name_LLM1 = "meta-llama/Meta-Llama-3-8B-Instruct"
model_LLM1 = AutoModelForCausalLM.from_pretrained(
    model_name_LLM1,
    load_in_8bit=False,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer_LLM1 = AutoTokenizer.from_pretrained(model_name_LLM1)

tokenizer_LLM1.add_special_tokens({"pad_token": "<PAD>"})
model_LLM1.resize_token_embeddings(len(tokenizer_LLM1))

sft_model_id_LLM1 = "./cp-task1-lora16-epoch5-llama3/final"
sft_model_LLM1: LoraModel = PeftModel.from_pretrained(
    model_LLM1,
    sft_model_id_LLM1,
    offload_folder="lora_results/lora_7/temp",
    is_trainable=True,
)
sft_model_LLM1 = sft_model_LLM1.merge_and_unload()

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

rl_model_LLM1 = AutoModelForCausalLMWithValueHead.from_pretrained(
    sft_model_LLM1, peft_config=lora_config
)


# Load model LLM_2:
model_name_LLM2 = "meta-llama/Meta-Llama-3-8B-Instruct"
model_LLM2 = AutoModelForCausalLM.from_pretrained(
    model_name_LLM2,
    load_in_8bit=False,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer_LLM2 = AutoTokenizer.from_pretrained(model_name_LLM2)

tokenizer_LLM2.add_special_tokens({"pad_token": "<PAD>"})
model_LLM2.resize_token_embeddings(len(tokenizer_LLM2))

sft_model_id_LLM2 = "./cp-task3-lora32-epoch5-llama3-fast/final"
sft_model_LLM2: LoraModel = PeftModel.from_pretrained(
    model_LLM2,
    sft_model_id_LLM2,
    offload_folder="lora_results/lora_7/temp",
    is_trainable=True,
)
sft_model_LLM2 = sft_model_LLM2.merge_and_unload()

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


rl_model_LLM2 = AutoModelForCausalLMWithValueHead.from_pretrained(
    sft_model_LLM2, peft_config=lora_config
)


# Need to add prompt template
def tokenize_1(sample):
    sample["input_ids"] = tokenizer_LLM1.encode(sample["description"])
    sample["query"] = tokenizer_LLM1.decode(sample["input_ids"])
    sample["label"] = sample["id"]
    return sample


def tokenize_2(sample):
    sample["input_ids"] = tokenizer_LLM2.encode(sample["input-prompt"])
    sample["query"] = tokenizer_LLM2.decode(sample["input_ids"])
    return sample


# Convert description to tokens!
ds = ds.map(tokenize_1, batched=False)
ds.set_format(type="torch")

generation_kwargs_llm1 = {
    "min_length": -1,
    "max_length": 1024,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer_LLM1.eos_token_id,
}

generation_kwargs_llm2 = {
    "min_length": -1,
    "max_length": 1024,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer_LLM2.eos_token_id,
}


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


ppo_config1 = PPOConfig(
    model_name="abcd",
    learning_rate=1.41e-5,
    log_with="wandb",
    batch_size=8,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
)

ppo_config2 = PPOConfig(
    model_name="abcd",
    learning_rate=1.41e-5,
    log_with="wandb",
    batch_size=8,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
)

ppo_trainer_LLM1 = PPOTrainer(
    config=ppo_config1,
    model=rl_model_LLM1,
    ref_model=None,
    tokenizer=tokenizer_LLM1,
    dataset=ds["train"],
    data_collator=collator,
)
ppo_trainer_LLM2 = PPOTrainer(
    config=ppo_config2, model=rl_model_LLM2, ref_model=None, tokenizer=tokenizer_LLM2
)

index = 0

# print the number of trainable parameters
print(
    f"Number of trainable parameters: {sum(p.numel() for p in ppo_trainer_LLM1.model.parameters() if p.requires_grad)}"
)
# print the percentage of trainable parameters
print(
    f"Percentage of trainable parameters: {sum(p.numel() for p in ppo_trainer_LLM1.model.parameters() if p.requires_grad) / sum(p.numel() for p in ppo_trainer_LLM1.model.parameters())}"
)

# Create a list to store the data
data = []
total_num = len(ppo_trainer_LLM1.dataloader)
print(f"=== Step: {index} / {total_num}) ===")
for epoch, batch in tqdm(enumerate(ppo_trainer_LLM1.dataloader)):
    # Tokenize input of llm1
    query_tensors_llm1: List[torch.Tensor] = batch["input_ids"]
    id_list: List[str] = batch["label"]

    # Get response from llm1
    response_tensors_llm1: List[torch.Tensor] = []
    for query in query_tensors_llm1:
        response = ppo_trainer_LLM1.generate(query, **generation_kwargs_llm1)
        response_tensors_llm1.append(response.squeeze())
    batch["response"] = [
        tokenizer_LLM1.decode(r.squeeze()) for r in response_tensors_llm1
    ]

    # extract code from the output
    texts_llm1: List[str] = [q + r for q, r in zip(batch["query"], batch["response"])]
    original_descriptions: List[str] = extract_desc(batch["query"])
    cleaned_codes: List[str] = extract_output(batch["response"])
    input_llm2: List[str] = prompt_task2(cleaned_codes)

    # Tokenize input of llm2
    query_tensors_llm2: List[torch.Tensor] = []
    for cleaned_code in input_llm2:
        sample = {"input-prompt": cleaned_code}
        sample = tokenize_2(sample)
        query_tensors_llm2.append(torch.tensor(sample["input_ids"]).to("cuda"))
    # Get response from llm2
    response_tensors_llm2: List[torch.Tensor] = []
    for query in query_tensors_llm2:
        # Reshape the query tensor
        query = query.reshape(1, -1)
        response = model_LLM2.generate(input_ids=query, **generation_kwargs_llm2)
        response_tensors_llm2.append(response.squeeze())
    response_llm2: List[str] = [
        tokenizer_LLM2.decode(r.squeeze()) for r in response_tensors_llm2
    ]

    # errors = check_error(cleaned_codes, id_list)
    regenerated_descriptions = extract_description_from_policy3(response_llm2)
    # Generate prompt for the input of LLM2:
    align_outputs = align_reward_func(regenerated_descriptions, original_descriptions)
    align_rewards = [torch.tensor(val) for val in align_outputs]

    pref_outputs_1 = pref_reward_func_1(
        reward_model_1, cleaned_codes, original_descriptions
    )
    preference_rewards_1 = [torch.tensor(val) for val in pref_outputs_1]

    #### Run PPO step
    rewards1 = [a for a in preference_rewards_1]
    stats1 = ppo_trainer_LLM1.step(query_tensors_llm1, response_tensors_llm1, rewards1)
    ppo_trainer_LLM1.log_stats(stats1, batch, rewards1)

    rewards2 = [b for b in align_rewards]
    stats2 = ppo_trainer_LLM2.step(query_tensors_llm2, response_tensors_llm2, rewards2)

    # Save in the dictionary:
    index += 1
    max_step = 63

    # Save the data in a dictionary
    for (
        id,
        original_desc,
        cleaned_code,
        regenerated_desc,
        reward1,
        llm1_response,
        llm2_response,
        reward2,
    ) in zip(
        batch["label"],
        original_descriptions,
        cleaned_codes,
        regenerated_descriptions,
        rewards1,
        batch["response"],
        response_llm2,
        rewards2,
    ):
        data.append(
            {
                "id": id,
                "Original Description": original_desc,
                "llm1 Response": llm1_response,
                "Cleaned Code": cleaned_code,
                "llm2 Response": llm2_response,
                "Regenerated Description": regenerated_desc,
                "Reward1": reward1.item(),
                "Reward2": reward2.item(),
            }
        )

        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        # Save the DataFrame as a CSV file
        os.makedirs("./evaluation", exist_ok=True)
        df.to_json(
            "./evaluation/output-joint-bert-pref-v3-2k-random-data.json",
            orient="records",
        )
    if index in [1, 63]:
        os.makedirs(
            f"./checkpoint/policy1-joint-pref-v3-2k-random-data-step{index}",
            exist_ok=True,
        )
        os.makedirs(
            f"./checkpoint/policy3-joint-bert-v3-2k-random-data-step{index}",
            exist_ok=True,
        )
        ppo_trainer_LLM1.save_pretrained(
            f"./checkpoint/policy1-joint-pref-v3-2k-random-data-step{index}"
        )
        ppo_trainer_LLM2.save_pretrained(
            f"./checkpoint/policy3-joint-bert-v3-2k-random-data-step{index}"
        )
    if index == max_step:
        break

    print(f"=== Step: ({index} / {total_num}) ===")
