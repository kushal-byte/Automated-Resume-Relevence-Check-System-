import re

def clean_text(text):
    """Remove extra spaces, line breaks, normalize text"""
    text = re.sub(r'\n+', '\n', text)   # collapse multiple newlines
    text = re.sub(r'\s+', ' ', text)    # collapse multiple spaces
    return text.strip()
