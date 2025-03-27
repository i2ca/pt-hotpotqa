from datasets import load_dataset
from translator import Translator
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
   
hotpot_qa = load_dataset("hotpot_qa", "distractor")
dataset = hotpot_qa['validation']

print(dataset)

quit()

haha = json.dumps(dataset[57])

print(haha)

quit()

missing = {
    "id": 0,
    "question": 0,
    "answer": 0,
    "supporting_facts": 0,
    "context": 0
}
for i, row in enumerate(dataset):
    if "id" in row:
        missing["id"] += 1
    if "question" in row:
        missing["question"] += 1
    if "supporting_facts" in row:
        missing["supporting_facts"] += 1
    if "context" in row:
        missing["context"] += 1

print(missing)
