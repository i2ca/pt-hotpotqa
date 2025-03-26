import re
import string
   
class StringNormalizer():

    @staticmethod
    def normalize(input: str) -> str:    
        normalized_input = input
        normalized_input = normalized_input.lower()
        normalized_input = normalized_input.split("</answer>")[0]
        normalized_input = normalized_input.translate(str.maketrans('', '', string.punctuation))
        normalized_input = ' '.join(normalized_input.split())
        #normalized_input = re.sub(r'\b(a|an|the)\b', ' ', normalized_input)

        return normalized_input
