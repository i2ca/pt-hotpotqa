import json
from datasets import load_dataset
from translator import Translator
#import uuid

def main():

    #uuid_string = str(uuid.uuid4())
    #print(uuid_string)

    temp_path = "./temp/temp_pt_hotpot_qa.json"
    output_path = "./datasets/pt_hotpot_qa_distractor_validation_v1.json"
    max_rows = 3

    hotpot_qa = load_dataset("hotpot_qa", "distractor")
    dataset = hotpot_qa['validation']
    num_completions = 0
    translator = Translator()

    with open(temp_path, "a", encoding="utf-8") as temp_file:
        for i, row in enumerate(dataset):
            if(i >= max_rows):
                break

            print(f"Translating row {i}")

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


if __name__ == "__main__":
    main()
