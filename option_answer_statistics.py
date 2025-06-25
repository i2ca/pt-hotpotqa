import os
import json
from llm_local import LlmLocalApi
from qa_solver_en import QaSolverEnglish
from qa_solver_pt import QaSolverPortuguese

class QADatasetStatistics:

    folder_path = "./evaluations/"

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
            self.qa_list = json.loads(file_str)

        n_correct = 0
        n_incorrect = 0
        for item in self.qa_list:
            if (item['correct']):
                n_correct += 1
            else:
                n_incorrect += 1

        n_total = n_correct + n_incorrect
        print(f'Correct: {n_correct}')
        print(f'Incorrect: {n_incorrect}')
        print(f'Total: {n_total}')
        print(f'Percent Correct: {100 * n_correct / n_incorrect:.2f}%')


