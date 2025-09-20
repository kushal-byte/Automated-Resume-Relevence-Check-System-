import fitz  # PyMuPDF
import pdfplumber
import docx

def extract_text_pymupdf(file_path):
    """Extract text from PDF using PyMuPDF"""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_pdfplumber(file_path):
    """Extract text from PDF using pdfplumber"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_docx(file_path):
    """Extract text from DOCX using python-docx"""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text
