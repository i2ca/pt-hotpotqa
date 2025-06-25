import os
import json
from llm_local import LlmLocalApi
from qa_solver_en import QaSolverEnglish
from qa_solver_pt import QaSolverPortuguese

class AnswerQADataset:

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

        print("Are the questions in this dataset in Portuguese or English?")
        print("    0. English")
        print("    1. Portuguese")
        language = int(input())

        print("Choose the LLM to answer this dataset")
        print("    0. Llama 1B local")
        print("    1. Llama 3B local")
        print("    2. Llama 1B bedrock")
        print("    3. Llama 3B bedrock")
        print("    4. Llama 11B bedrock")
        print("    5. Llama 70B bedrock")
        print("    6. Chat GPT-4o")
        self.llm_number = int(input())

        if (self.llm_number == 0):
            llmApi = LlmLocalApi(model="meta-llama/Llama-3.2-1B-Instruct")
        if (self.llm_number == 1):
            llmApi = LlmLocalApi(model="meta-llama/Llama-3.2-3B-Instruct")

        print("Enter the name of the file for the answers to be saved.")
        output_filename = input()

        temp_path = "./evaluations/" + output_filename + ".temp.json"
        error_path = "./evaluations/" + output_filename + ".errors.json"
        output_path = "./evaluations/" + output_filename + ".json"

        if (language == 0):
            qa_solver = QaSolverEnglish(llmApi)
        else:
            qa_solver = QaSolverPortuguese(llmApi)
        qa_solver.answer_dataset(self.question_list, temp_path, error_path, output_path)

