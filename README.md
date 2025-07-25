# Llama-3.2-3B-Computer-Engineering-LLM

--- 

A custom fine-tuned large language model based on Meta's LLaMA 3.2 3B, specialized for computer engineering applications. This model has been fine-tuned on high-quality datasets including Wikitext-2-raw-v1, STEM-AI-mtl for Electrical Engineering, and additional computer science and computer engineering data. Fine-tuning was performed using LoRA (Low-Rank Adaptation) adapters to efficiently inject domain-specific knowledge into the model while preserving its general language understanding.

---

<div align="center">
  <a href="https://github.com/IrfanUruchi/Llama-3.2-3B-Computer-Engineering-LLM">
    <img src="https://img.shields.io/badge/🔗_GitHub-Repo-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <a href="https://huggingface.co/Irfanuruchi/Llama-3.2-3B-Computer-Engineering-LLM">
    <img src="https://img.shields.io/badge/🤗_HuggingFace-Model_Repo-FFD21F?style=for-the-badge" alt="HuggingFace">
  </a>
  <br>
  <img src="https://img.shields.io/badge/Model_Size-3.2B_parameters-blue" alt="Model Size">
  <img src="https://img.shields.io/badge/Quantization-4bit-green" alt="Quantization">
  <img src="https://img.shields.io/badge/Adapter-LoRA-orange" alt="Adapter">
  <img src="https://img.shields.io/badge/Context-8k-lightgrey" alt="Context">
  <img src="https://img.shields.io/badge/License-Llama_3.2-yellow" alt="License">
</div>
---

## Model Details

- **Base Model:** Meta's LLaMA 3.2 3B
- **Architecture:** LlamaForCausalLM
- **Quantization:** Loaded in 8-bit mode using BitsAndBytes

---

## Installation

If you have Hugging Face installed using python:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "Irfanuruchi/Llama-3.2-3B-Computer-Engineering-LLM",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# System prompt template
system_prompt = """[INST] <<SYS>>
You are an expert computer engineer with specialization in:
- Computer architecture
- Embedded systems design
- Hardware-software co-design
<</SYS>>"""

# Query example
user_query = """Explain the concept of pipelining in CPU design:"""
inputs = tokenizer(system_prompt + user_query, return_tensors="pt").to("cuda")

# Generation config
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using GIT

If you want to install it using Git feel free to use the following tutorial:



Ensure you have Git LFS installed for handling large model files:

```bash

git lfs install

```

**Usage**

Clone the repository and pull the model files using Git LFS(for pro users):

```bash


pip install -U bitsandbytes transformers torch accelerate
```
Or just download the files one by one(if you have hard time with LFS).


Then load the model locally a simple sample test how you can do it 

```python

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

local_path = "./Llama-3.2-3B-Computer-Engineering-LLM" #Make sure you are in the same directory as the downloaded model and config files else will give error
model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto", torch_dtype=torch.float16, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False, local_files_only=True)

# An example , this is a prompt to use the LLM after laoding 
prompt = "What is a kernel?"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.92)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
---

## Fine-tuning details

Datesets i used for fine tuning: Wikitext-2-raw-v1, STEM-AI-mtl (Electrical Engineering), and additional computer science and computer engineering data.

For fine tuning was done with LoRA adapters

---

## Model file 
The model is provided as a safetensors file and/or compressed as a 7z archive (1.9GB)
Download the model directly via LFS or by downloading the file from repo.

---

## License and Attribution

This model is a derivative work based on Meta's LLaMA 3.2 3B and is distributed under the LLaMA 3.2 Community License. Please see the LICENSE file for full details.

Attribution:
“Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Built with Llama.”

For more information on the base model, please visit :

https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE

---

## Known Limitations

The model is specialized for computer engineering and electical engineering topics and may not work as well on unrelated subjects, some outputs may require further prompt engineering for optimal results.
And occasional repetition or other artifacts may be present due to fine-tuning constraints.




