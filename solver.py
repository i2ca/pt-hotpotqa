from datasets import load_dataset
from bedrock_api import BedrockApi
   
hotpot_qa = load_dataset("hotpot_qa", "distractor")
dataset = hotpot_qa['validation']



item = dataset[0]
supporting_facts = item["supporting_facts"]

context = item["context"]

full_context = ""

for title, sentences in zip(context['title'], context['sentences']):
    full_context += f"{title}\n"
    for sentence in sentences:
        full_context += f"  {sentence.strip()}"
    full_context += "\n\n"


question = item["question"]

prompt = f"""\
<prompt>
Considering the following context and question, answer the question with only the answer. No explanation is necessary.
</prompt>
<context>
{full_context}
</context>
<question>
{question}
</question>
<answer>
"""

print(item["answer"])


llmApi = BedrockApi()
completion = llmApi.call_llama_api(prompt)

print(completion)