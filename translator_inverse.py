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
        prompt = f"""\
Translate only the text values (e.g., inside quotes) from the following JSON object from Portuguese to English.
Do not change the structure, keys, punctuation, or formatting of the JSON.
Do not replace double quotes " with single quotes '.
Return the result as valid JSON
{text}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            n=1,
            temperature=0.0
        )
        translated_text = response.choices[0].message.content.strip()
        return translated_text

    def translate_json(self, json_dict):
        input_string = json.dumps(json_dict, ensure_ascii=False)
        translated_string = self.translate_text(input_string)
        translated_string = translated_string.removeprefix("```json").removesuffix("```")

        print("translated_string -----------------------------------------------------")
        print(translated_string)

        output_json_dict = {} 
        try:
            output_json_dict = json.loads(translated_string)
        except json.JSONDecodeError as e:
            None
            #print(f"Error decoding completion JSON: json_dict: {input_string}.")
        return output_json_dict