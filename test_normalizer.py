import re
import string
from string_normalizer import StringNormalizer
   


completion = "U.S.  A."

normalized_completion = StringNormalizer.normalize(completion)

print(f"{normalized_completion}")
