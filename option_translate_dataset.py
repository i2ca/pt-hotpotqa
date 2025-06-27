import os
import json
from llm_openai import LlmOpenaiApi
from qa_translator_en_pt import QaTranslatorEnglishToPortuguese
from qa_translator_pt_en import QaTranslatorPortugueseToEnglish

class TranslateQADataset:

    folder_path = "./datasets/"

    def run(self):

        files = []
        for f in os.listdir(self.folder_path):
            if os.path.isfile(os.path.join(self.folder_path, f)):
                files.append(f)

        print("Choose a dataset:")
        for i, file_name in enumerate(files):
            print(f"    {i}. {file_name}")

        print("Type the number of the dataset you want:")
        self.file_number = int(input())

        file_path = os.path.join(self.folder_path, files[self.file_number])

        with open(file_path, 'r', encoding='utf-8') as pt_file:
            file_str = pt_file.read()
            self.question_list = json.loads(file_str)

        print("Please choose the languages to translate:")
        print("    0. English to Portuguese")
        print("    1. Portuguese to English")
        language = int(input())

        llmApi = LlmOpenaiApi()

        if (language == 0):
            qa_translator = QaTranslatorEnglishToPortuguese(llmApi)
        if (language == 1):
            qa_translator = QaTranslatorPortugueseToEnglish(llmApi)

        print("Enter the name of the file for the answers to be saved.")
        output_filename = input()

        temp_path = "./evaluations/" + output_filename + ".temp.json"
        error_path = "./evaluations/" + output_filename + ".errors.json"
        output_path = "./evaluations/" + output_filename + ".json"

        qa_translator.translate_dataset(self.question_list, temp_path, error_path)

