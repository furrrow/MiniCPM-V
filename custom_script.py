"""
custom script for running MiniCPM-V 2.6
https://huggingface.co/openbmb/MiniCPM-V-2_6

Inference using Huggingface transformers on NVIDIA GPUs. Requirements tested on python 3.10：

Pillow==10.1.0
torch==2.1.2
torchvision==0.16.2
transformers==4.40.0
sentencepiece==0.1.99
decord
numpy < 2
pip install flash-attn --no-build-isolation
"""

# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
access_token = "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"
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
                                  token=access_token, attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

image = Image.open('xx.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    stream=True
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
