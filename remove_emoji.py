import re
import os
import sys

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)
        
    # Remove markdown and formatting characters
    formatting_pattern = re.compile(r'[*_~`#]|\[|\]|\(\)|```|`|>|#+')
    text = formatting_pattern.sub('', text)
    
    # Remove multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
    #return emoji_pattern.sub(r'', text)

mystr="Hello! How can I assist you today? 😊. Here are points 1. **Apoorva**  2. *Pascal* "
print(mystr)
print(remove_emojis(mystr))

