import os
import json
import time
from datasets import Dataset
from bedrock_api import BedrockApi
from string_normalizer import StringNormalizer

dataset_path = "./datasets/x100_pt_hotpot_qa.json"

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

subset = Dataset.from_list(data)

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
Dado o contexto abaixo, responda à pergunta de forma concisa, fornecendo apenas a resposta direta, sem explicações ou justificativas adicionais.
</prompt>
<context>
A Torre Eiffel está localizada na cidade de Paris, que é a capital da França. Ela foi construída em 1889.
</context>
<question>
Em que cidade está localizada a Torre Eiffel?
</question>
<answer>
Paris
</answer>
<prompt>
Dado o contexto abaixo, responda à pergunta de forma concisa, fornecendo apenas a resposta direta, sem explicações ou justificativas adicionais.
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
        time.sleep(30)