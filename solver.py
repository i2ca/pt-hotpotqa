import random
import os
import json
import time
from datasets import load_dataset
from bedrock_api import BedrockApi
from string_normalizer import StringNormalizer
  
hotpot_qa = load_dataset("hotpot_qa", "distractor")
dataset = hotpot_qa['validation']

print(dataset)

seed = 853350
shuffled = dataset.shuffle(seed=seed)
subset = shuffled.select(range(100))

temp_path = "./temp/temp_solution.json"
id_set = set()
if os.path.exists(temp_path):
    print("A temporary file exists, the process will continue where it stopped.")
    with open(temp_path, 'r', encoding='utf-8') as file:
        temp_file_str = file.read()
        temp_file_str = "[" + temp_file_str.rstrip(",\n") + "]"
        temp_list = json.loads(temp_file_str)
        for temp_element in temp_list:
            id_set.add(temp_element["id"])
    print(f"Entries recovered from the temporary file: {len(id_set)}")

n_correct = 0
n_incorrect = 0

with open(temp_path, 'a', encoding='utf-8') as temp_file:
    for i, item in enumerate(subset):
        if i >= 100:
            break

        if item["id"] in id_set:
            print(f"Already solved {i}: {item['id']}")
            continue

        supporting_facts = item["supporting_facts"]
        context = item["context"]
        full_context = ""
        for title, sentences in zip(context['title'], context['sentences']):
            full_context += f"{title}\n"
            for sentence in sentences:
                full_context += f"  {sentence.strip()}"
            full_context += "\n\n"
        question = item["question"]
        answer = item["answer"]
        prompt = f"""\
<prompt>
Given the context below, answer the question concisely, providing only the direct answer without additional explanations or justifications.
</prompt>
<context>
The Eiffel Tower is located in the city of Paris, which is the capital of France. It was constructed in 1889.
</context>
<question>
In which city is the Eiffel Tower located?
</question>
<answer>
Paris
</answer>
<prompt>
Given the context below, answer the question concisely, providing only the direct answer without additional explanations or justifications.
</prompt>
<context>
{full_context}
</context>
<question>
{question}
</question>
<answer>
"""

        llmApi = BedrockApi()
        completion = llmApi.call_llama_api(prompt)

        normalized_completion = StringNormalizer.normalize(completion)
        normalized_answer = StringNormalizer.normalize(answer)

        correct = (normalized_completion == normalized_answer)

        if correct:
            n_correct += 1
        else:
            n_incorrect += 1

        data = {
            "id": item["id"],
            "level": item["level"],
            "type": item["type"],
            "prompt": prompt,
            "completion": completion,
            "answer": answer,
            "normalized_completion": normalized_completion,
            "normalized_answer": normalized_answer,
            "correct": correct
        }

        json.dump(data, temp_file, ensure_ascii=False, indent=4)
        temp_file.write(",\n")
        temp_file.flush()
        id_set.add(item["id"])

        print(f"Completion={normalized_completion}, Answer={normalized_answer}, Correct={n_correct}, Incorrect={n_incorrect}, {item['type']}-{item['level']}")
        time.sleep(20)