import json
from qa_translator import QaTranslator
from llm_api import LlmApi

class QaTranslatorPortugueseToEnglish(QaTranslator):

    def __init__(self, llmApi: LlmApi):
        self.llmApi = llmApi

    #Translates a given text from English to Portuguese using GPT-4o-mini.
    def translate_text(self, text):

        prompt = f"""\
Translate only the text values (e.g., inside quotes) from the following JSON object from Portuguese to English.
Do not change the structure, keys, punctuation, or formatting of the JSON.
Do not replace double quotes " with single quotes '.
The result must be a valid JSON.
Return only the translated JSON object and nothing else.
{text}
"""     
        translated_text = self.llmApi.query(prompt)
        return translated_text