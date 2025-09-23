import os
from typing import Optional
import PyPDF2
from docx import Document as DocxDocument

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from DOCX: {str(e)}")

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from TXT: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error extracting text from TXT: {str(e)}")

def extract_text(file_path: str) -> tuple[str, str]:
    """Extract text from a file based on its extension.

    Returns:
        tuple: (extracted_text, file_type)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path).lower()

    if filename.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
        file_type = 'pdf'
    elif filename.endswith('.docx'):
        text = extract_text_from_docx(file_path)
        file_type = 'docx'
    elif filename.endswith('.txt'):
        text = extract_text_from_txt(file_path)
        file_type = 'txt'
    else:
        # Try to read as text file for other extensions
        try:
            text = extract_text_from_txt(file_path)
            file_type = 'txt'
        except:
            raise ValueError(f"Unsupported file type: {filename}")

    return text, file_type