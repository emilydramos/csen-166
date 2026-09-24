**Part 1**

```python
import math
squares = [1, 4, 9, 16, 25]
items = {headphones: 100, soap: 5, shirt: 15}

for i in range(5):
  print(i)
```

**Part 2**

Model #1

Link: https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF

Model #2: Transformers

Link: https://github.com/huggingface/transformers

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


Model #3:

Code to execute:
```python
from IPython.display import Audio


sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[0].numpy(), rate=sampling_rate)
```
**Part 3**
