from qa_solver import QaSolver
from llm_api import LlmApi
from string_normalizer import StringNormalizer
  
class QaSolverPortuguese(QaSolver):

    def __init__(self, llmApi: LlmApi):
        self.llmApi = llmApi

    def answer(self, question_object: str):
        item = question_object

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

        completion = self.llmApi.query(prompt)

        normalized_completion = StringNormalizer.normalize(completion)
        normalized_answer = StringNormalizer.normalize(answer)

        correct = (normalized_completion == normalized_answer)

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

        return data