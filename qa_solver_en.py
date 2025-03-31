from qa_solver import QaSolver
from llm_api import LlmApi
from string_normalizer import StringNormalizer
  
class QaSolverEnglish(QaSolver):

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