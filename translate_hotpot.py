import os
import time
import json
from datasets import load_dataset
from translator import Translator
#import uuid

def main():

    start_time = time.time()  # Record the start time

    #uuid_string = str(uuid.uuid4())
    #print(uuid_string)

    temp_path = "./temp/temp_pt_hotpot_qa.json"
    output_path = "./datasets/pt_hotpot_qa_distractor_validation_v1.json"
    max_rows = 3

    hotpot_qa = load_dataset("hotpot_qa", "distractor")
    dataset = hotpot_qa['validation']
    num_completions = 0
    translator = Translator()


    id_set = set()

    if os.path.exists(temp_path):
        print("A temporary file exists, the process will continue where it stopped.")
        with open(temp_path, 'r', encoding='utf-8') as file:
            temp_file_str = file.read()
            temp_file_str = "[" + temp_file_str.rstrip(",\n") + "]"
            temp_list = json.loads(temp_file_str)
            for temp_element in temp_list:
                id_set.add(temp_element["id"])
        print(f"Entries recovered from the temporary file: {len(id_set)}")

    with open(temp_path, "a", encoding="utf-8") as temp_file:
        for i, row in enumerate(dataset):
            if(i >= max_rows):
                print()
                print(f"Max number of rows achieved (max={max_rows}).")
                break

            if row["id"] in id_set:
                print(f"\rAlready translated {i}: {row["id"]}", end="\r")
                continue

            print(f"Translating row {i}: {row["id"]}", end="\r")

            to_translate = {
                'question': row['question'],
                'answer': row['answer'],
                'supporting_facts': row['supporting_facts'],
                'context': row['context']
            }

            translated_row = translator.translate_json(to_translate)

            output_row = {
                'id': row['id'],
                'question': translated_row['question'],
                'answer': translated_row['answer'],
                'type': row['type'],
                'level': row['level'],
                'supporting_facts': translated_row['supporting_facts'],
                'context': translated_row['context']
            }

            json.dump(output_row, temp_file, ensure_ascii=False, indent=4)
            temp_file.writelines(",\n")
            num_completions += 1
            temp_file.flush()

        print()

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Time spent in execution: {elapsed_time:.3f} seconds.")


if __name__ == "__main__":
    main()
