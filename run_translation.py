from datasets import load_dataset
from llm_local import LlmLocalApi
from llm_openai import LlmOpenaiApi
from qa_translator_en_pt import QaTranslatorEnglishToPortuguese

temp_path = "./temp/teste_3.json"

hotpot_qa = load_dataset("hotpot_qa", "distractor")
dataset = hotpot_qa['train']
subset = dataset.shuffle(seed=42).select(range(10))

llm_api = LlmOpenaiApi()

qa_translator = QaTranslatorEnglishToPortuguese(llm_api)

output = qa_translator.translate_dataset(subset, temp_path="./temp/a3.json", error_path="./errors/e3.json")

