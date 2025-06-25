from option_answer_dataset import AnswerQADataset
from option_answer_statistics import QADatasetStatistics

folder_path = "./datasets/"  # current directory, or change to your desired folder

print()
print("Choose an option:")

print("    0. Translate a QA dataset")
print("    1. Use an LLM to answer a QA dataset")
print("    2. Show a QA answer dataset stats")
option = input()

if option == "1":
    opt1 = AnswerQADataset()
    opt1.run()

if option == "2":
    opt2 = QADatasetStatistics()
    opt2.run()


