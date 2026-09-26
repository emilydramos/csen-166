**Part 1**

```python
import math
squares = [1, 4, 9, 16, 25]
inventory = {headphones: 100, soap: 5, shirt: 15}

for i in range(5):
  print(i)
class Item{
  def __init__(self, name; str, price: float = 0.0)
  self.name = name
  price = price
}
```

**Part 2**

Model #1
Link: https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF

input 1:

input 2:

input 3: 

Model #2: Qwen3.5-0.8B

Link: https://huggingface.co/Qwen/Qwen3.5-0.8B#qwen35-08b

description: This model can handle text, images, videos, etc.

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

output: The best way to live a good life is 12 hours a day, 7 days a week, and it is not possible to live a


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


Model #3: MusicGen
Link: https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/musicgen#musicgen
description: This model can generate audio samples (based on text prompts as well). 

Code to execute:
```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", device_map="auto")

inputs = processor(
    text=["[INPUT HERE]"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
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



**Part 3**


**Part 4: Google Colab**

Tested the models in pt. 2 in Google Colab. 


**Part 5: Reflection ** 

