**Part 1 - Basic Python**

```python
import math
squares = [1, 4, 9, 16, 25]
name = "Kyle"


for i in range(5):
  print(i)

inventory = {"headphones": 100, "soap": 5, "shirt": 15}
for key, value in inventory.items():
    print(f"Key: {key}, Value: {value}")

class Item:
  def __init__(self, name: str, price: float = 0.0):
    self.name = name
    self.price = price

  
item1 = Item(name="ball", price=15)
print(item1.name)
```

**Part 2 - Testing Models**

Model #1
Link: https://huggingface.co/Salesforce/blip-image-captioning-large

description: This model can perform a variety of vision-language tasks. It can be used for both conditional (user prompt)/unconditional (without user prompt) image captioning.

code to execute: 

```python
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large") 
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

img_url = '[IMAGE URL HERE]' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB') # convert to rgb format for processor

# conditional image captioning
text = "a photo of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16) 

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True)) # strips out formatting markers

```

input 1: https://unsplash.com/photos/three-iced-drinks-on-wooden-table-pJxT7UgdvkI

output: a photo of two people holding cups of drinks on a table

input 2: https://images.unsplash.com/photo-1790207504527-0dd91be5d7f5?q=80&w=928&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D

output: a photo of a carnival with a ferris wheel and people walking around

input 3: https://plus.unsplash.com/premium_vector-1788874552998-c2c07a58791f?q=80&w=872&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D

output: a photo of a pixel style picture of two rings on a blue background

Model #2: Qwen3.5-0.8B

Link: https://huggingface.co/Qwen/Qwen3.5-0.8B#qwen35-08b

description: This model can handle text, images, videos, etc and is meant for small, low-power tasks. In this case, we are using it for autocompletion/text generation. 

Code to execute:

```python

import torch
from transformers import pipeline

pipe = pipeline( 
    task="text-generation",
    model="Qwen/Qwen3.5-0.8B", 
    device_map="auto",
)
print(pipe("[INPUT HERE], ", max_new_tokens=20)[0]["generated_text"])

```

input 1:

```python
 pipeline("The best way to live a good life is ")
```

output 1: The best way to live a good life is 12 hours a day, 7 days a week, and it is not possible to live a


input 2:

```python
print(pipe("The capital of California is ", max_new_tokens=20)[0]["generated_text"])
```

output: The capital of California is 1.5 miles from the Los Angeles area and 1.5 miles from the San Diego area


input 3 (I set the max tokens here to 50)

```python
print(pipe("Once upon a time, in a land far, far away, ", max_new_tokens=50)[0]["generated_text"])
```

output: Once upon a time, in a land far, far away, 1500 miles from your home, there lived a wise old man named Mr. Wisdom. He was known for his wisdom, but he was also very much into the world. One day, he decided to meet with a group of friends to


observations: Instead of ending a sentence at a natural point, the model would sometimes cut off in the middle of the sentence. It seems to want to reach the token limit. 

Model #3: MusicGen
Link: https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/musicgen#musicgen
description: This model can generate audio samples (based on text prompts as well). 

Code to execute:
```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration 

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", device_map="auto") #selects model type, automatically allocates model weights based on available memory

inputs = processor(
    text=["[INPUT HERE]"],
    padding=True,
    return_tensors="pt", # returns tensors
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256) # guidance scale determines how closely the generated clip will pertain to the prompt


```

to listen to the output:
```python
from IPython.display import Audio # need to import library 

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```
input 1:
```python
 text=["80s upbeat synth track"],
```
output: 5 second 80s upbeat synth track

input 2:
```python
 text=["calm and soothing piano for studying"],
```
output: 5 second calm piano track

input 3:
```python
 text=["drum and bass track"],
```

output: 5 second d&b track, no visible melody 
note: I also tried the same prompt with tokens = 500, which changed the output to about 7 seconds instead of 5. 

observation: The model worked well with a variety of different genre inputs. It was slow to run on Colab. 

**Part 3 - WAVE**

job to test:
```python
 cart_price = 0
for i in range(6):
  cart_price += 4

print(cart_price)
```

evidence:

<img width="804" height="378" alt="Wave job" src="https://github.com/user-attachments/assets/80827dc3-cb96-4d5b-afd8-7edc5a83ccec" />


process to run:
1. Open https://scu-ood.wave.scu.edu/pun/sys/dashboard
2. Under Interactive Apps, click Launch JupyterLab
3. Select appropriate amount of resources and version of Python
4. Job will be queued and when it's ready to connect, there will be a button in Jupyter Notebook. 

Resources used: 1 CPU, 8 GB, 3 hours with Python version 3.13 (no GPUs). 

**Part 4: Google Colab**

Tested all of the models in pt. 2 in Google Colab. 

```python

import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

img_url = 'https://plus.unsplash.com/premium_vector-1788874552998-c2c07a58791f?q=80&w=872&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "a photo of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
```

Resources used: 6.1 GB of system RAM, 1.0 of GPU RAM, 44.2 GB on Disk.

evidence:

<img width="1311" height="419" alt="Screenshot 2026-10-01 at 7 12 23 PM" src="https://github.com/user-attachments/assets/4ad8e65e-47f5-4c52-9654-27028a6fe1ae" />

**Part 5: Reflection ** 

I found Google Colab an intuitive, simple way to test models that were not as resource-intensive. Although it will not be feasible for larger datasets and models, it is more available. The technical challenges I expect to have are with WAVE, and being able to remember what to put in the batch script, remembering to deallocate resources when finished, etc. I think all the resources we've covered in this lab will be useful for the team project, whether they will be used as the first way of testing or as a backup plan. 
