import docx2txt

def extract_text_docx(file_path):
    """Extract text from DOCX"""
    return docx2txt.process(file_path)
