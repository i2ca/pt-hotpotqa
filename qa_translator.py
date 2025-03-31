import os
import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from validator import HotpotEntryValidator
  
class QaTranslator(ABC):

    @abstractmethod
    def translate_text(self, question_object):
        pass

    def translate_json(self, json_dict):
        input_string = json.dumps(json_dict, ensure_ascii=False)
        translated_string = self.translate_text(input_string)
        translated_string = translated_string.removeprefix("```json").removesuffix("```")

        output_json_dict = {}
        error_obj={}
        try:
            output_json_dict = json.loads(translated_string)
        except json.JSONDecodeError as e:
            error_obj = {
                "has_error": True,
                "completion": translated_string,
                "error_messages": [e.msg],
            }

        if error_obj:
            return error_obj

        return output_json_dict

    def qa_translate(self, row):
        to_translate = {
            'question': row['question'],
            'answer': row['answer'],
            'supporting_facts': row['supporting_facts'],
            'context': row['context']
        }
        
        translation = self.translate_json(to_translate)

        if "has_error" in translation and translation["has_error"]:
            return translation

        error_messages = HotpotEntryValidator.validate(translation)
        if error_messages:
            return {
                "has_error": True,
                "completion": translation,
                "error_messages": error_messages,
            }

        translated_row = {
            'id': row['id'],
            'question': translation['question'],
            'answer': translation['answer'],
            'type': row['type'],
            'level': row['level'],
            'supporting_facts': {
                'title': translation['supporting_facts']['title'],
                'sent_id': row['supporting_facts']['sent_id']
            },
            'context': translation['context']
        }

        return translated_row

    def translate_dataset(self, dataset, temp_path, error_path) -> str:
        start_time = time.time()  # Record the start time
        
        max_rows = 8000
        num_threads = 100  # Number of concurrent threads
        
        translation_list = []
        id_set = set()

        if os.path.exists(temp_path):
            print("A temporary file exists, the process will continue where it stopped.")
            with open(temp_path, 'r', encoding='utf-8') as file:
                temp_file_str = file.read()
                temp_file_str = "[" + temp_file_str.rstrip(",\n") + "]"
                translation_list = json.loads(temp_file_str)
                for temp_element in translation_list:
                    id_set.add(temp_element["id"])
            print(f"Entries recovered from the temporary file: Total={len(translation_list)}, Unique={len(id_set)}")

        n_completed = 0

        with \
        ThreadPoolExecutor(max_workers=num_threads) as executor, \
        open(temp_path, "a", encoding="utf-8") as temp_file, \
        open(error_path, "a", encoding="utf-8") as error_file:
            futures = {}
            started_threads = 0
            for i, row in enumerate(dataset):
                if started_threads >= num_threads:
                    print("Waiting API to return the results.")
                    n_success = 0
                    n_error = 0
                    for future in as_completed(futures):
                        translated_row = future.result()
                        if "has_error" in translated_row and translated_row["has_error"]:
                            json.dump(translated_row, error_file, ensure_ascii=False, indent=4)
                            error_file.write(",\n")
                            error_file.flush()
                            n_error += 1
                        else:
                            translation_list.append(translated_row)
                            json.dump(translated_row, temp_file, ensure_ascii=False, indent=4)
                            temp_file.write(",\n")
                            temp_file.flush()
                            id_set.add(translated_row["id"])
                            n_success += 1
                        n_completed += 1
                    started_threads = 0
                    futures = {}
                    print(f"Partial results saved in temporary file. n_success={n_success} / n_error={n_error}")

                if i >= max_rows:
                    print(f"Max number of rows achieved (max={max_rows}).")
                    break

                if row["id"] in id_set:
                    print(f"Already translated {i}: {row['id']}")
                    continue
                
                print(f"Submitting row {i}: {row['id']} for translation")
                future = executor.submit(self.qa_translate, row)
                futures[future] = row["id"]
                started_threads += 1

            print("Waiting API to return the last batch of results.")
            n_success = 0
            n_error = 0
            for future in as_completed(futures):
                translated_row = future.result()
                if "has_error" in translated_row and translated_row["has_error"]:
                    json.dump(translated_row, error_file, ensure_ascii=False, indent=4)
                    error_file.write(",\n")
                    error_file.flush()
                    n_error += 1
                else:
                    translation_list.append(translated_row)
                    json.dump(translated_row, temp_file, ensure_ascii=False, indent=4)
                    temp_file.write(",\n")
                    temp_file.flush()
                    id_set.add(translated_row["id"])
                    n_success += 1
                n_completed += 1
            started_threads = 0
            futures = {}
            print(f"Partial results saved in temporary file. n_success={n_success} / n_error={n_error}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Time spent in execution: {elapsed_time:.3f} seconds.")

        return translation_list
