import os
import json
    
temp_path = "./temp/temp_solution_11b_vision.json"

if os.path.exists(temp_path):
    print("A temporary file exists, the process will continue where it stopped.")

with open(temp_path, 'r', encoding='utf-8') as file:
    temp_file_str = file.read()
    temp_file_str = "[" + temp_file_str.rstrip(",\n") + "]"
    temp_list = json.loads(temp_file_str)

missing = {
    "id": 0
}

n_correct = 0
for element in temp_list:
    if "id" in element:
        missing["id"] += 1
    if element["correct"]:
        n_correct += 1



print(len(temp_list))
print(n_correct)

