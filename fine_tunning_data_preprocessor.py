import os
import json
from datasets import Dataset
    
dataset_path = "./datasets/pt_hotpot_qa_distractor_train.json"
output_path = "./datasets/fine_tunning"

if os.path.exists(dataset_path):
    print("File exists in this path")

with open(dataset_path, 'r', encoding='utf-8') as file:
    temp_file_str = file.read()
    dataset_list = json.loads(temp_file_str)

list_text = []

for i, item in enumerate(dataset_list):
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

    text = f"""\
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
{answer}
<answer>
"""
    list_text.append(text)

dataset = Dataset.from_dict({"text": list_text})
dataset.save_to_disk(output_path)