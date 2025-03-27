import os
import json
    
temp_path = "./datasets/x100_pt_hotpot_qa.json"

if os.path.exists(temp_path):
    print("A temporary file exists, the process will continue where it stopped.")

with open(temp_path, 'r', encoding='utf-8') as file:
    temp_file_str = file.read()
    temp_list = json.loads(temp_file_str)


id_set = set()

for i, element in enumerate(temp_list):
    if element["id"] in id_set:
        print(i)
        print(element["id"])
    id_set.add(element["id"])

print(len(temp_list))
print(len(id_set))

