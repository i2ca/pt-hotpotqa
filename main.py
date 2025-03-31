from datasets import load_dataset
from llm_local import LlmLocalApi
from qa_solver_en import QaSolverEnglish


temp_path = "./temp/teste_2.json"

hotpot_qa = load_dataset("hotpot_qa", "distractor")
dataset = hotpot_qa['validation']

llm_api = LlmLocalApi()

qa_solver = QaSolverEnglish(llm_api)

qa_solver.answer_dataset(dataset, temp_path)
