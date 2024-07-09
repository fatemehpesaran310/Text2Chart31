# Text2Chart31
Official PyTorch implementation of "Text2Chart31: Instruction Tuning for Chart Generation with Automatic Feedback"

![image samples](asset/figure.png)

> **Abstract** *Large language models (LLMs) have demonstrated strong capabilities across various language tasks, notably through instruction-tuning methods. However, LLMs face challenges in visualizing complex, real-world data through charts and plots. Firstly, existing datasets rarely cover a full range of chart types, such as 3D, volumetric, and gridded charts. Secondly, supervised fine-tuning methods do not fully leverage the intricate relationships within rich datasets, including text, code, and figures. To address these challenges, we propose a hierarchical pipeline and a new dataset for chart generation. Our dataset, Text2Chart31, includes 31 unique plot types referring to the Matplotlib library, with 11.1K tuples of descriptions, code, data tables, and plots. Moreover, we introduce a reinforcement learning-based instruction tuning technique for chart generation tasks without requiring human feedback. Our experiments show that this approach significantly enhances the model performance, enabling smaller models to outperform larger open-source models and be comparable to state-of-the-art proprietary models in data visualization tasks.*

## Dataset File
- Code, Description, Data Table and Reasoning Step: 
    - Training set: `./prepare-data/Text2Chart-31-train.json`
    - Test set: `./prepare-data/Text2Chart-31-test.json`.
- Dataset file including the figures: 
    - Training set: [download](https://drive.google.com/file/d/11otHdVt7eJqAJ7RJl71G6eFBsKHNYEAM/view?usp=sharing)
    - Test set: [download](https://drive.google.com/file/d/1ckNEhhWA-eGPiGl-j7Mc_5UldsNNtOZX/view?usp=sharing)


## LoRA checkpoints
Unzip it under `checkpoint` folder and run inference code.
### Supervised fine-tuned model

| Task  | Model | Checkpoints |
| :------ | :------ | :------: |
| Task 1 | Llama 3 Instruct | [download](https://drive.google.com/file/d/1DfG4kHO1N4QeG5SMVlVqMpqFQBNr3qhr/view?usp=sharing) |
| Task 2 | Llama 3 Instruct | TBA |
| Task 3 | Llama 3 Instruct | [download](https://drive.google.com/file/d/14Yyju22AXbQ_lakOkzt_eMkv7YN2iKyc/view?usp=sharing) |


## Reward model checkpoint
- OPT model: [download](https://drive.google.com/file/d/1W7HsPs4F2Js1l8zO-iCNRiXLcML4AswP/view?usp=sharing)

## Training code

### Supervised fine-tuning
- Task 1: Run `python sft-task1.py`
- Task 2: Run `python sft-task2.py`
- Task 3: Run `python sft-task3.py`

### RL fine-tuning
- Task 1 & Task 3: Run `python rl-task1-task3.py` (You would need to download reward model/SFT model checkpoints beforehand).

## Generating samples

### Task 1
- Base model : Run `python generate-llama3-base.py`
- SFT model : Run `python generate-llama3-bf16-sft.py`
- RL model : Run `python generate-llama3-bf16-rl.py`

### Task 2
- SFT model : Run `python generate2-llama3-sft.py` (You would need to train the model beforehand).

### Task 3
- Base model : Run `python generate3-llama3-base.py`
- SFT model : Run `python generate3-llama3-bf16-sft.py`
- RL model : Run `python generate3-llama3-bf16-rl.py`

