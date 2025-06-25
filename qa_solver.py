import os
import json
from abc import ABC, abstractmethod
  
class QaSolver(ABC):

    @abstractmethod
    def answer(self, question_object) -> str:
        pass

    def answer_dataset(self, dataset, temp_path, error_path, output_path) -> str:
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

        with open(temp_path, 'a', encoding='utf-8') as temp_file:
            for i, item in enumerate(dataset):
                if item["id"] in id_set:
                    print(f"Already solved {i}: {item['id']}")
                    continue

                qa_object = self.answer(item)

                json.dump(qa_object, temp_file, ensure_ascii=False, indent=4)
                temp_file.write(",\n")
                temp_file.flush()
                id_set.add(item["id"])

                print(f"{qa_object['id']}")
                #time.sleep(20)

        with open(temp_path, 'r', encoding='utf-8') as file:
            temp_file_str = file.read()
            temp_file_str = "[" + temp_file_str.rstrip(",\n") + "]"
            final_list = json.loads(temp_file_str)

        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(final_list, output_file)