# pdf_processor.py
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

def calculate_pdf_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of PDF content for deduplication"""
    return hashlib.sha256(file_content).hexdigest()

def validate_pdf_content(file_content: bytes) -> None:
    """Validate PDF content before processing"""
    if not file_content:
        raise PDFProcessingError("PDF content is empty")
    
    if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
        raise PDFProcessingError("PDF file too large (max 50MB)")
    
    # Check if it's actually a PDF by looking for PDF signature
    if not file_content.startswith(b'%PDF-'):
        raise PDFProcessingError("Invalid PDF format")

def extract_text_from_pdf(file_content: bytes) -> List[Dict[str, any]]:
    """Extract text from PDF with improved error handling"""
    validate_pdf_content(file_content)
    
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to open PDF document: {e}")
        raise PDFProcessingError(f"Could not open PDF: {str(e)}")
    
    docs_with_pages = []
    total_chars = 0
    
    try:
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text().strip()
                
                if text:
                    # Filter out very short chunks that are likely noise
                    if len(text) >= 10:
                        docs_with_pages.append({
                            "page_content": text,
                            "metadata": {
                                "page": page_num + 1,
                                "char_count": len(text)
                            }
                        })
                        total_chars += len(text)
                        
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                continue
                
    finally:
        doc.close()
    
    if not docs_with_pages:
        raise PDFProcessingError("No readable text found in PDF")
    
    if total_chars < 50:
        raise PDFProcessingError("PDF contains insufficient text content")
    
    logger.info(f"Extracted text from {len(docs_with_pages)} pages, {total_chars} total characters")
    return docs_with_pages

def create_text_chunks(docs_with_pages: List[Dict[str, any]], 
                      chunk_size: int = 1000, 
                      chunk_overlap: int = 100) -> List[Dict[str, any]]:
    """Create text chunks with improved chunking strategy"""
    
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size")
    
    # Adjust chunk size based on content complexity
    avg_page_length = sum(len(doc["page_content"]) for doc in docs_with_pages) / len(docs_with_pages)
    
    if avg_page_length < 500:
        # For documents with short pages, use smaller chunks
        chunk_size = min(chunk_size, 500)
        chunk_overlap = min(chunk_overlap, 50)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separation hierarchy
    )
    
    split_chunks = []
    chunk_id = 0
    
    for doc_page in docs_with_pages:
        page_content = doc_page["page_content"]
        page_number = doc_page["metadata"]["page"]
        
        try:
            page_splits = text_splitter.create_documents([page_content])
            
            for split in page_splits:
                chunk_text = split.page_content.strip()
                
                # Skip very short chunks
                if len(chunk_text) < 20:
                    continue
                
                split_chunks.append({
                    "text": chunk_text,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split())
                })
                chunk_id += 1
                
        except Exception as e:
            logger.warning(f"Error splitting text for page {page_number}: {e}")
            continue
    
    if not split_chunks:
        raise PDFProcessingError("Failed to create any valid text chunks")
    
    logger.info(f"Created {len(split_chunks)} text chunks")
    return split_chunks

def load_and_split_pdf(file_content: bytes, 
                      chunk_size: int = 1000, 
                      chunk_overlap: int = 100) -> List[Dict[str, any]]:
    """
    Main function to load PDF and split into chunks with comprehensive error handling
    
    Args:
        file_content: PDF file content as bytes
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of dictionaries containing text chunks and metadata
        
    Raises:
        PDFProcessingError: If PDF processing fails
    """
    try:
        # Extract text from PDF
        docs_with_pages = extract_text_from_pdf(file_content)
        
        # Create text chunks
        chunks = create_text_chunks(docs_with_pages, chunk_size, chunk_overlap)
        
        # Add PDF metadata
        pdf_hash = calculate_pdf_hash(file_content)
        for chunk in chunks:
            chunk["pdf_hash"] = pdf_hash
            chunk["processing_timestamp"] = None  # Can be set by calling function
        
        return chunks
        
    except PDFProcessingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in PDF processing: {e}", exc_info=True)
        raise PDFProcessingError(f"PDF processing failed: {str(e)}")

# Utility functions for PDF analysis
def analyze_pdf_content(file_content: bytes) -> Dict[str, any]:
    """Analyze PDF content and return metadata"""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        
        metadata = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "file_size": len(file_content),
            "pdf_hash": calculate_pdf_hash(file_content)
        }
        
        # Analyze text density
        total_chars = 0
        pages_with_text = 0
        
        for page_num in range(len(doc)):
            try:
                text = doc[page_num].get_text()
                if text.strip():
                    total_chars += len(text)
                    pages_with_text += 1
            except Exception:
                continue
        
        metadata.update({
            "total_characters": total_chars,
            "pages_with_text": pages_with_text,
            "avg_chars_per_page": total_chars / max(pages_with_text, 1),
            "text_density": pages_with_text / len(doc) if len(doc) > 0 else 0
        })
        
        doc.close()
        return metadata
        
    except Exception as e:
        logger.error(f"Error analyzing PDF content: {e}")
        return {"error": str(e)}