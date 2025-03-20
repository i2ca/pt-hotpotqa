from openai import OpenAI
import json

class Translator:
    def __init__(self, api_key_file="api_key.txt"):
        self.api_key = self.get_api_key(api_key_file)
        self.client = OpenAI(api_key=self.api_key)

    def get_api_key(self, file_path):
        with open(file_path, "r") as file:
            api_key = file.readline().rstrip('\n')
        return api_key

    #Translates a given text from English to Portuguese using GPT-4o-mini.
    def translate_text(self, text):
        prompt = f"Translate the following text from English to Portuguese:\n\n{text}"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            n=1
        )
        translated_text = response.choices[0].message.content.strip()
        return translated_text

    def translate_json(self, json_dict):
        input_string = json.dumps(json_dict, ensure_ascii=False)
        translated_string = self.translate_text(input_string)
        output_json_dict = {} 
        try:
            output_json_dict = json.loads(translated_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding completion JSON: {json_dict["id"]}.")
            return {}
        return output_json_dict