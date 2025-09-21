# parsers/universal_parser.py - Universal Resume Parser
import os
import re
from pathlib import Path

class UniversalResumeParser:
    """Universal parser that handles multiple resume formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._extract_from_pdf,
            '.docx': self._extract_from_docx,
            '.txt': self._extract_from_txt,
            '.doc': self._extract_from_doc
        }
        print("✅ Universal Resume Parser initialized")
    
    def extract_text(self, file_path):
        """Extract text from any supported file format"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext not in self.supported_formats:
                # Fallback to text reading
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except:
                    raise ValueError(f"Unsupported format: {file_ext}")
            
            print(f"🔍 Processing {file_ext} file...")
            
            # Use appropriate extractor
            extractor = self.supported_formats[file_ext]
            text = extractor(file_path)
            
            # Clean text
            enhanced_text = self._enhance_extracted_text(text)
            
            print(f"✅ Extracted {len(enhanced_text)} characters")
            return enhanced_text
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            # Try basic text reading as fallback
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except:
                return f"Error extracting from {file_path}: {str(e)}"
    
    def _extract_from_pdf(self, file_path):
        """Extract from PDF using existing function"""
        try:
            from parsers.pdf_parser import extract_text_pymupdf
            return extract_text_pymupdf(file_path)
        except ImportError:
            # Fallback if PyMuPDF not available
            try:
                import fitz
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except ImportError:
                return "PDF extraction requires PyMuPDF package"
        except Exception as e:
            return f"PDF extraction error: {str(e)}"
    
    def _extract_from_docx(self, file_path):
        """Extract from DOCX using existing function"""
        try:
            from parsers.docx_parser import extract_text_docx
            return extract_text_docx(file_path)
        except ImportError:
            try:
                import docx
                doc = docx.Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                return "DOCX extraction requires python-docx package"
        except Exception as e:
            return f"DOCX extraction error: {str(e)}"
    
    def _extract_from_txt(self, file_path):
        """Extract from text file with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                return raw_data.decode('utf-8', errors='ignore')
        except Exception as e:
            return f"Text extraction error: {str(e)}"
    
    def _extract_from_doc(self, file_path):
        """Extract from legacy DOC format"""
        try:
            import docx2txt
            text = docx2txt.process(file_path)
            return text
        except ImportError:
            return "DOC format requires docx2txt package (pip install docx2txt)"
        except Exception as e:
            return f"DOC extraction error: {str(e)}"
    
    def _enhance_extracted_text(self, text):
        """Clean and enhance extracted text"""
        if not text or len(text.strip()) < 10:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common extraction issues
        text = re.sub(r'([a-zA-Z0-9._%+-]+)\s*@\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'\1@\2', text)
        text = re.sub(r'(\d{3})\s*-?\s*(\d{3})\s*-?\s*(\d{4})', r'\1-\2-\3', text)
        
        return text.strip()

def test_universal_parser():
    """Test the universal parser"""
    parser = UniversalResumeParser()
    test_text = "Test resume text"
    enhanced = parser._enhance_extracted_text(test_text)
    print("✅ Universal parser test completed")
    return True

if __name__ == "__main__":
    test_universal_parser()
