#service/pdf_processor
import fitz  # PyMuPDF
import pdfplumber
import tabula
import camelot
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional, Tuple, Any
import logging
import hashlib
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TableExtractionMethod(Enum):
    TABULA = "tabula"
    CAMELOT = "camelot"
    PDFPLUMBER = "pdfplumber"

class DocumentType(Enum):
    COURSE_CATALOG = "course_catalog"
    SCHEDULE = "schedule"
    GRADES = "grades"
    RESEARCH_PAPER = "research_paper"
    GENERAL = "general"

@dataclass
class ExtractedTable:
    data: List[List[str]]
    headers: Optional[List[str]]
    page_number: int
    table_index: int
    extraction_method: str
    confidence_score: float
    table_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProcessedDocument:
    text_chunks: List[Dict[str, Any]]
    tables: List[ExtractedTable]
    document_metadata: Dict[str, Any]
    document_type: DocumentType

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

class EnhancedPDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Patterns for detecting different document types
        self.catalog_patterns = [
            r"semester\s*\d+",
            r"credits?\s*:?\s*\d+",
            r"hours?\s*:?\s*\d+",
            r"instructor|teacher|professor",
            r"course\s*code",
            r"subject\s*code"
        ]
    
    def detect_document_type(self, text_content: str) -> DocumentType:
        """Detect the type of document based on content patterns"""
        text_lower = text_content.lower()
        
        catalog_score = sum(1 for pattern in self.catalog_patterns 
                          if re.search(pattern, text_lower, re.IGNORECASE))
        
        if catalog_score >= 3:
            return DocumentType.COURSE_CATALOG
        
        return DocumentType.GENERAL
    
    def extract_tables_with_tabula(self, file_content: bytes) -> List[ExtractedTable]:
        """Extract tables using tabula-py"""
        tables = []
        try:
            # Extract tables from all pages
            df_list = tabula.read_pdf(
                file_content,
                pages='all',
                multiple_tables=True,
                pandas_options={'header': None},
                silent=True
            )
            
            for page_idx, df in enumerate(df_list):
                if df.empty:
                    continue
                    
                # Convert DataFrame to list of lists
                table_data = []
                headers = None
                
                # Try to detect headers
                if not df.iloc[0].isna().all():
                    potential_headers = df.iloc[0].astype(str).tolist()
                    if any(header.strip() for header in potential_headers):
                        headers = potential_headers
                        table_data = df.iloc[1:].fillna('').astype(str).values.tolist()
                    else:
                        table_data = df.fillna('').astype(str).values.tolist()
                else:
                    table_data = df.fillna('').astype(str).values.tolist()
                
                # Calculate confidence score based on data density
                total_cells = len(table_data) * len(table_data[0]) if table_data else 0
                filled_cells = sum(1 for row in table_data for cell in row if cell.strip())
                confidence = filled_cells / total_cells if total_cells > 0 else 0
                
                tables.append(ExtractedTable(
                    data=table_data,
                    headers=headers,
                    page_number=page_idx + 1,  # Approximate page number
                    table_index=0,
                    extraction_method=TableExtractionMethod.TABULA.value,
                    confidence_score=confidence
                ))
                
        except Exception as e:
            logger.warning(f"Tabula table extraction failed: {e}")
        
        return tables
    
    def extract_tables_with_camelot(self, file_content: bytes) -> List[ExtractedTable]:
        """Extract tables using camelot"""
        tables = []
        try:
            # Save content temporarily for camelot
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file.flush()
                
                try:
                    # Extract tables with camelot
                    camelot_tables = camelot.read_pdf(tmp_file.name, pages='all')
                    
                    for table in camelot_tables:
                        df = table.df
                        
                        # Convert to our format
                        table_data = df.fillna('').astype(str).values.tolist()
                        headers = None
                        
                        # Try to detect headers
                        if not df.iloc[0].isna().all():
                            potential_headers = df.iloc[0].astype(str).tolist()
                            if any(header.strip() for header in potential_headers):
                                headers = potential_headers
                                table_data = df.iloc[1:].fillna('').astype(str).values.tolist()
                        
                        tables.append(ExtractedTable(
                            data=table_data,
                            headers=headers,
                            page_number=table.page,
                            table_index=0,
                            extraction_method=TableExtractionMethod.CAMELOT.value,
                            confidence_score=table.accuracy / 100.0,  # Camelot provides accuracy
                            metadata={'parsing_report': table.parsing_report}
                        ))
                        
                finally:
                    os.unlink(tmp_file.name)
                    
        except Exception as e:
            logger.warning(f"Camelot table extraction failed: {e}")
        
        return tables
    
    def extract_tables_with_pdfplumber(self, file_content: bytes) -> List[ExtractedTable]:
        """Extract tables using pdfplumber"""
        tables = []
        try:
            with pdfplumber.open(file_content) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_idx, table_data in enumerate(page_tables):
                        if not table_data:
                            continue
                        
                        # Clean the table data
                        cleaned_data = []
                        headers = None
                        
                        for row_idx, row in enumerate(table_data):
                            cleaned_row = [cell.strip() if cell else '' for cell in row]
                            
                            # First non-empty row might be headers
                            if row_idx == 0 and any(cell for cell in cleaned_row):
                                # Check if this looks like headers
                                if any(re.search(r'[a-zA-Z]', cell) for cell in cleaned_row):
                                    headers = cleaned_row
                                else:
                                    cleaned_data.append(cleaned_row)
                            else:
                                cleaned_data.append(cleaned_row)
                        
                        if cleaned_data:
                            # Calculate confidence
                            total_cells = len(cleaned_data) * len(cleaned_data[0])
                            filled_cells = sum(1 for row in cleaned_data for cell in row if cell.strip())
                            confidence = filled_cells / total_cells if total_cells > 0 else 0
                            
                            tables.append(ExtractedTable(
                                data=cleaned_data,
                                headers=headers,
                                page_number=page_num + 1,
                                table_index=table_idx,
                                extraction_method=TableExtractionMethod.PDFPLUMBER.value,
                                confidence_score=confidence
                            ))
                            
        except Exception as e:
            logger.warning(f"PDFPlumber table extraction failed: {e}")
        
        return tables
    
    def merge_and_deduplicate_tables(self, all_tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """Merge tables from different extraction methods and remove duplicates"""
        if not all_tables:
            return []
        
        # Group tables by page
        page_tables = {}
        for table in all_tables:
            page_num = table.page_number
            if page_num not in page_tables:
                page_tables[page_num] = []
            page_tables[page_num].append(table)
        
        final_tables = []
        
        for page_num, tables in page_tables.items():
            if not tables:
                continue
            
            # If multiple extraction methods found tables on same page, pick best one
            if len(tables) > 1:
                # Sort by confidence score and data richness
                tables.sort(key=lambda t: (t.confidence_score, len(t.data)), reverse=True)
                
                # Take the best table, but check for significant differences
                best_table = tables[0]
                
                # Check if other tables have significantly different structure
                for other_table in tables[1:]:
                    if (abs(len(other_table.data) - len(best_table.data)) > 2 or
                        abs(len(other_table.data[0]) - len(best_table.data[0])) > 1):
                        # Significantly different table, keep both
                        final_tables.append(other_table)
                
                final_tables.append(best_table)
            else:
                final_tables.append(tables[0])
        
        return final_tables
    
    def classify_table_type(self, table: ExtractedTable, document_type: DocumentType) -> str:
        """Classify the type of table based on content"""
        if document_type != DocumentType.COURSE_CATALOG:
            return "general"
        
        # Check headers and content for catalog-specific patterns
        all_text = []
        if table.headers:
            all_text.extend(table.headers)
        
        for row in table.data[:3]:  # Check first few rows
            all_text.extend(row)
        
        content_text = ' '.join(all_text).lower()
        
        catalog_indicators = [
            r'subject|course',
            r'teacher|instructor|professor',
            r'credits?',
            r'hours?',
            r'semester'
        ]
        
        score = sum(1 for pattern in catalog_indicators 
                   if re.search(pattern, content_text, re.IGNORECASE))
        
        if score >= 3:
            return "course_catalog"
        elif score >= 1:
            return "academic_schedule"
        
        return "general"
    
    def process_pdf_comprehensive(self, file_content: bytes) -> ProcessedDocument:
        """Main function to process PDF with both text and table extraction"""
        
        # Validate PDF
        self.validate_pdf_content(file_content)
        
        # Extract basic text content
        docs_with_pages = self.extract_text_from_pdf(file_content)
        
        # Detect document type
        all_text = ' '.join([doc["page_content"] for doc in docs_with_pages])
        document_type = self.detect_document_type(all_text)
        
        # Create text chunks
        text_chunks = self.create_text_chunks(docs_with_pages)
        
        # Extract tables using multiple methods
        logger.info("Extracting tables using multiple methods...")
        
        tabula_tables = self.extract_tables_with_tabula(file_content)
        camelot_tables = self.extract_tables_with_camelot(file_content)
        pdfplumber_tables = self.extract_tables_with_pdfplumber(file_content)
        
        # Merge and deduplicate
        all_tables = tabula_tables + camelot_tables + pdfplumber_tables
        final_tables = self.merge_and_deduplicate_tables(all_tables)
        
        # Classify table types
        for table in final_tables:
            table.table_type = self.classify_table_type(table, document_type)
        
        logger.info(f"Extracted {len(final_tables)} tables and {len(text_chunks)} text chunks")
        
        # Document metadata
        document_metadata = self.analyze_pdf_content(file_content)
        document_metadata.update({
            'document_type': document_type.value,
            'table_count': len(final_tables),
            'text_chunk_count': len(text_chunks)
        })
        
        return ProcessedDocument(
            text_chunks=text_chunks,
            tables=final_tables,
            document_metadata=document_metadata,
            document_type=document_type
        )
    
    # Keep existing methods from your current pdf_processor.py
    def validate_pdf_content(self, file_content: bytes) -> None:
        """Validate PDF content before processing"""
        if not file_content:
            raise PDFProcessingError("PDF content is empty")
        
        if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
            raise PDFProcessingError("PDF file too large (max 50MB)")
        
        if not file_content.startswith(b'%PDF-'):
            raise PDFProcessingError("Invalid PDF format")

    def extract_text_from_pdf(self, file_content: bytes) -> List[Dict[str, any]]:
        """Extract text from PDF with improved error handling"""
        self.validate_pdf_content(file_content)
        
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

    def create_text_chunks(self, docs_with_pages: List[Dict[str, any]], 
                          chunk_size: int = 1000, 
                          chunk_overlap: int = 100) -> List[Dict[str, any]]:
        """Create text chunks with improved chunking strategy"""
        
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        split_chunks = []
        chunk_id = 0
        
        for doc_page in docs_with_pages:
            page_content = doc_page["page_content"]
            page_number = doc_page["metadata"]["page"]
            
            try:
                page_splits = self.text_splitter.create_documents([page_content])
                
                for split in page_splits:
                    chunk_text = split.page_content.strip()
                    
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

    def analyze_pdf_content(self, file_content: bytes) -> Dict[str, any]:
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
                "pdf_hash": hashlib.sha256(file_content).hexdigest()
            }
            
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

# Create global instance
enhanced_pdf_processor = EnhancedPDFProcessor()

# Backward compatibility functions
def load_and_split_pdf(file_content: bytes, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, any]]:
    """Backward compatibility wrapper"""
    processed = enhanced_pdf_processor.process_pdf_comprehensive(file_content)
    return processed.text_chunks

def analyze_pdf_content(file_content: bytes) -> Dict[str, any]:
    """Backward compatibility wrapper"""
    return enhanced_pdf_processor.analyze_pdf_content(file_content)