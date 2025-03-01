# Llama-3.2-3B-Computer-Engineering-LLM
A custom fine-tuned large language model based on Meta's LLaMA 3.2 3B, specialized for computer engineering applications. This model has been fine-tuned on high-quality datasets including Wikitext-2-raw-v1, STEM-AI-mtl for Electrical Engineering, and additional computer science and computer engineering data. Fine-tuning was performed using LoRA (Low-Rank Adaptation) adapters to efficiently inject domain-specific knowledge into the model while preserving its general language understanding.

## Model Details

- **Base Model:** Meta's LLaMA 3.2 3B
- **Architecture:** LlamaForCausalLM
- **Quantization:** Loaded in 8-bit mode using BitsAndBytes

## Installation

Ensure you have Git LFS installed for handling large model files:

```bash

git lfs install

pip install -U bitsandbytes transformers torch accelerate
```

## Usage

Clone the repository and pull the model files using Git LFS(for more pro users):

```bash
git lfs install
pip install -U bitsandbytes transformers torch accelerate
```
Or just download the files one by one.


Then load the model locally 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
local_path = "./Llama-3.2-3B-Computer-Engineering-LLM" #Make sure you are in the same directory as the downloaded model and config files else will give error

model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto", torch_dtype=torch.float16, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False, local_files_only=True)

# An example of a usage , this is a prompt
prompt = "What is a kernel?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8, top_k=50, top_p=0.92)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## Fine-tuning details

Datesets i used for fine tuning: Wikitext-2-raw-v1, STEM-AI-mtl (Electrical Engineering), and additional computer science and computer engineering data.

For fine tuning was done with LoRA adapters

## Model file 
The model is provided as a safetensors file and/or compressed as a 7z archive (1.9GB)
Download the model directly via LFS or by downloading the file from repo.


## License and Attribution

This model is a derivative work based on Meta's LLaMA 3.2 3B and is distributed under the LLaMA 3.2 Community License. Please see the LICENSE file for full details.

Attribution:
“Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. Built with Llama.”

For more information on the base model, please visit :

https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE

## Known Limitations

The model is specialized for computer engineering and electical engineering topics and may not work as well on unrelated subjects, some outputs may require further prompt engineering for optimal results.
And occasional repetition or other artifacts may be present due to fine-tuning constraints.




