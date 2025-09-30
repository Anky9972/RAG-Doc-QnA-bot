# app/services/llamacloud_processor.py

import logging
import time
import hashlib
import os
import tempfile
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from functools import wraps

# Try multiple import paths for LlamaParse
try:
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    try:
        from llama_cloud import LlamaParse
        LLAMAPARSE_AVAILABLE = True
    except ImportError:
        try:
            from llama_cloud_services import LlamaParse
            LLAMAPARSE_AVAILABLE = True
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.warning("LlamaParse not found. Please install with: pip install llama-parse")
            LLAMAPARSE_AVAILABLE = False
            LlamaParse = None

from app.core import config
from app.schemas import ProcessedDocument, DocumentType, TextChunk, ExtractedTable

logger = logging.getLogger(__name__)


class LlamaCloudError(Exception):
    """Custom exception for LlamaCloud processing errors"""
    def __init__(self, message: str, error_code: str = None, original_error: Exception = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        super().__init__(self.message)


@dataclass
class LlamaParseResult:
    """Result from LlamaParse processing"""
    text_content: str
    markdown_content: str
    image_documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float


def retry_on_failure(max_retries=3, delay=1.0):
    """Decorator for retrying failed operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


class LlamaCloudProcessor:
    """
    Enhanced PDF processor using LlamaCloud/LlamaParse for complex document processing
    """
    
    def __init__(self):
        self.api_key = None
        self.parser = None
        self._cache: Dict[str, ProcessedDocument] = {}
        self._initialized = False
        
        if LLAMAPARSE_AVAILABLE:
            try:
                self.api_key = self._get_api_key()
                self._initialize_parser()
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize LlamaCloud processor: {e}")
                self._initialized = False
        else:
            logger.warning("LlamaParse not available - skipping initialization")
            self._initialized = False
        
    def _get_api_key(self) -> str:
        """Get LlamaCloud API key from environment or config"""
        # Try environment variable first
        api_key = os.getenv('LLAMA_CLOUD_API_KEY')
        
        # Try config if available
        if not api_key and hasattr(config, 'LLAMA_CLOUD_API_KEY'):
            api_key = config.LLAMA_CLOUD_API_KEY
            
        if not api_key:
            raise LlamaCloudError(
                "LLAMA_CLOUD_API_KEY not found. Please set it in environment variables.\n"
                "Get your API key from: https://cloud.llamaindex.ai/",
                error_code="MISSING_API_KEY"
            )
            
        # Validate API key format
        if not api_key.startswith('llx-'):
            logger.warning("API key doesn't match expected format (should start with 'llx-')")
            
        return api_key
    
    def _initialize_parser(self):
        """Initialize LlamaParse with correct parameters"""
        if not LLAMAPARSE_AVAILABLE:
            raise LlamaCloudError("LlamaParse package not available", error_code="PACKAGE_NOT_FOUND")
            
        try:
            self.parser = LlamaParse(
                api_key=self.api_key,
                num_workers=4,
                verbose=True,
                language="en"
            )
            logger.info("LlamaParse initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaParse: {e}")
            raise LlamaCloudError(
                f"Parser initialization failed: {str(e)}", 
                error_code="INIT_FAILED",
                original_error=e
            )
    
    def is_available(self) -> bool:
        """Check if the processor is properly initialized and available"""
        return self._initialized and self.parser is not None
    
    def _validate_pdf_input(self, pdf_content: bytes, filename: str):
        """Validate PDF input before processing"""
        if not pdf_content:
            raise LlamaCloudError("PDF content is empty", error_code="EMPTY_CONTENT")
        
        # Check file size (100MB limit)
        if len(pdf_content) > 100 * 1024 * 1024:
            raise LlamaCloudError(
                "PDF file too large (>100MB)", 
                error_code="FILE_TOO_LARGE"
            )
        
        if not filename.lower().endswith('.pdf'):
            logger.warning(f"File {filename} doesn't have .pdf extension")
        
        # Check if content is actually PDF
        if not pdf_content.startswith(b'%PDF'):
            raise LlamaCloudError(
                "File doesn't appear to be a valid PDF", 
                error_code="INVALID_PDF"
            )
    
    def _get_cache_key(self, pdf_content: bytes) -> str:
        """Generate cache key for PDF content"""
        return hashlib.sha256(pdf_content).hexdigest()
    
    def process_pdf_comprehensive(self, pdf_content: bytes, filename: str = "document.pdf") -> ProcessedDocument:
        """
        Process PDF using LlamaCloud with comprehensive extraction
        
        Args:
            pdf_content: Raw PDF bytes
            filename: Original filename for context
            
        Returns:
            ProcessedDocument with enhanced extraction results
        """
        if not self.is_available():
            raise LlamaCloudError(
                "LlamaCloud processor not available - check initialization", 
                error_code="NOT_AVAILABLE"
            )
        
        # Validate input
        self._validate_pdf_input(pdf_content, filename)
        
        # Check cache first
        cache_key = self._get_cache_key(pdf_content)
        if cache_key in self._cache:
            logger.info(f"Returning cached result for {filename}")
            return self._cache[cache_key]
        
        start_time = time.time()
        tmp_file_path = None
        
        try:
            logger.info(f"Starting LlamaCloud processing for {filename}")
            
            # Save to temporary file (LlamaParse needs file path)
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file_path = tmp_file.name
            
            # Parse with LlamaCloud
            llama_result = self._parse_with_llamacloud(tmp_file_path)
            
            # Convert to our ProcessedDocument format
            processed_doc = self._convert_to_processed_document(
                llama_result, pdf_content, filename
            )
            
            # Cache the result
            self._cache[cache_key] = processed_doc
            
            processing_time = time.time() - start_time
            logger.info(f"LlamaCloud processing completed in {processing_time:.2f}s")
            
            return processed_doc
                
        except Exception as e:
            logger.error(f"LlamaCloud processing failed for {filename}: {e}")
            if isinstance(e, LlamaCloudError):
                raise
            raise LlamaCloudError(
                f"Processing failed: {str(e)}", 
                error_code="PROCESSING_FAILED",
                original_error=e
            )
        finally:
            # Always clean up temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    logger.debug(f"Cleaned up temp file: {tmp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {tmp_file_path}: {e}")
    
    @retry_on_failure(max_retries=3)
    def _parse_with_llamacloud(self, file_path: str) -> LlamaParseResult:
        """Parse document using LlamaCloud with the correct API"""
        start_time = time.time()
        
        try:
            # Parse the document
            result = self.parser.parse(file_path)
            
            # Extract content from pages using the correct API
            text_content = ""
            markdown_content = ""
            processed_images = []
            pages_info = []
            
            # Handle the result based on its structure
            if hasattr(result, 'pages') and result.pages:
                # New API structure with pages
                for page_idx, page in enumerate(result.pages):
                    if hasattr(page, 'text'):
                        text_content += page.text + "\n\n"
                    if hasattr(page, 'md'):
                        markdown_content += page.md + "\n\n"
                    
                    # Process images if available
                    if hasattr(page, 'images') and page.images:
                        for img_idx, img in enumerate(page.images):
                            processed_images.append({
                                'page_number': page_idx + 1,
                                'image_index': img_idx,
                                'description': getattr(img, 'description', getattr(img, 'text', '')),
                                'image_type': getattr(img, 'type', getattr(img, 'image_type', 'unknown')),
                                'size': getattr(img, 'size', 0)
                            })
                    
                    # Page metadata
                    pages_info.append({
                        'page_number': page_idx + 1,
                        'has_images': hasattr(page, 'images') and bool(page.images),
                        'has_layout': hasattr(page, 'layout') and bool(page.layout),
                        'has_structured_data': hasattr(page, 'structuredData') and bool(page.structuredData)
                    })
            
            elif isinstance(result, list):
                # Handle list of documents (older API or batch processing)
                for doc_idx, doc in enumerate(result):
                    if hasattr(doc, 'text'):
                        text_content += doc.text + "\n\n"
                    if hasattr(doc, 'get_content'):
                        markdown_content += doc.get_content() + "\n\n"
            
            else:
                # Handle single document result
                if hasattr(result, 'text'):
                    text_content = result.text
                if hasattr(result, 'get_content'):
                    markdown_content = result.get_content()
                elif hasattr(result, 'content'):
                    markdown_content = result.content
            
            # If we didn't get markdown content, use text content
            if not markdown_content and text_content:
                markdown_content = text_content
            
            # Create metadata
            metadata = {
                'parsing_time': time.time() - start_time,
                'num_pages': len(pages_info) if pages_info else 1,
                'num_images': len(processed_images),
                'parser_version': 'llamacloud',
                'language': 'en',
                'pages_info': pages_info
            }
            
            return LlamaParseResult(
                text_content=text_content,
                markdown_content=markdown_content,
                image_documents=processed_images,
                metadata=metadata,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"LlamaParse processing failed: {e}")
            raise LlamaCloudError(
                f"LlamaParse failed: {str(e)}", 
                error_code="PARSE_FAILED",
                original_error=e
            )
    
    def _convert_to_processed_document(
        self, 
        llama_result: LlamaParseResult, 
        pdf_content: bytes, 
        filename: str
    ) -> ProcessedDocument:
        """Convert LlamaParseResult to ProcessedDocument format"""
        
        # Generate file hash
        pdf_hash = hashlib.sha256(pdf_content).hexdigest()
        
        # Create text chunks from markdown (better structure preservation)
        text_chunks = self._create_text_chunks(
            llama_result.markdown_content or llama_result.text_content, 
            llama_result.metadata.get('pages_info', [])
        )
        
        # Extract tables from structured content
        tables = self._extract_tables_from_markdown(llama_result.markdown_content or llama_result.text_content)
        
        # Create document metadata
        document_metadata = {
            'filename': filename,
            'file_size': len(pdf_content),
            'pdf_hash': pdf_hash,
            'page_count': llama_result.metadata.get('num_pages', 0),
            'processing_method': 'llamacloud',
            'processing_time': llama_result.processing_time,
            'extraction_quality': 'high',
            'has_images': llama_result.metadata.get('num_images', 0) > 0,
            'has_tables': len(tables) > 0,
            'language': llama_result.metadata.get('language', 'en'),
            'images_info': llama_result.image_documents,
            'llamacloud_metadata': llama_result.metadata
        }
        
        # Determine document type based on content analysis
        document_type = self._determine_document_type(
            llama_result.markdown_content or llama_result.text_content, 
            tables
        )
        
        return ProcessedDocument(
            text_chunks=text_chunks,
            tables=tables,
            document_metadata=document_metadata,
            document_type=document_type,
            processing_method='llamacloud'
        )
    
    def _create_text_chunks(self, content: str, pages_info: List[Dict[str, Any]]) -> List[TextChunk]:
        """Create text chunks from markdown content with improved chunking"""
        chunks = []
        
        if not content:
            return chunks
        
        # Split by double newlines and headers
        sections = self._smart_content_split(content)
        
        chunk_id = 0
        current_page = 1
        
        for section in sections:
            if not section.strip():
                continue
                
            # Estimate page number from content position
            page_number = self._estimate_page_number(section, pages_info, current_page)
            
            # Create chunk with metadata
            chunk = TextChunk(
                id=chunk_id,
                text=section.strip(),
                page_number=page_number,
                chunk_type='markdown_section',
                metadata={
                    'extraction_method': 'llamacloud',
                    'content_type': self._classify_content_type(section),
                    'word_count': len(section.split()),
                    'has_table': '|' in section and section.count('|') > 3,
                    'has_list': any(line.strip().startswith(('- ', '* ', '1. ', '2. ')) for line in section.split('\n')),
                    'header_level': self._get_header_level(section)
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            if page_number > current_page:
                current_page = page_number
        
        logger.info(f"Created {len(chunks)} text chunks from LlamaCloud content")
        return chunks
    
    def _smart_content_split(self, content: str, max_chunk_size: int = 2000) -> List[str]:
        """Smart content splitting that preserves structure"""
        sections = []
        
        if not content:
            return sections
        
        # Split by headers first
        lines = content.split('\n')
        current_section = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            # Check if this is a header
            is_header = (
                line.strip().startswith('#') or 
                (line.strip() and len(line.strip()) < 100 and 
                 not line.strip().endswith('.') and
                 not line.strip().startswith('|') and  # Not a table line
                 not re.match(r'^\s*[-*+]\s', line))  # Not a list item
            )
            
            # If we hit a header and have content, save current section
            if is_header and current_section and current_size > 500:
                sections.append('\n'.join(current_section))
                current_section = [line]
                current_size = line_size
            
            # If current section is getting too large, split it
            elif current_size + line_size > max_chunk_size and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
                current_size = line_size
            
            else:
                current_section.append(line)
                current_size += line_size
        
        # Add remaining content
        if current_section:
            sections.append('\n'.join(current_section))
        
        return [s for s in sections if s.strip()]  # Remove empty sections
    
    def _classify_content_type(self, content: str) -> str:
        """Classify the type of content in a chunk"""
        content_lower = content.lower()
        content_stripped = content.strip()
        
        if '|' in content and content.count('|') > 5:
            return 'table'
        elif content_stripped.startswith('#'):
            return 'header'
        elif any(content_stripped.startswith(prefix) for prefix in ['- ', '* ', '1. ', '2. ']):
            return 'list'
        elif any(keyword in content_lower for keyword in ['figure', 'image', 'chart', 'graph', 'diagram']):
            return 'figure_description'
        elif len(content.split()) > 100:
            return 'paragraph'
        else:
            return 'text'
    
    def _get_header_level(self, content: str) -> int:
        """Get the header level from markdown content"""
        first_line = content.split('\n')[0].strip()
        if first_line.startswith('#'):
            return first_line.count('#')
        return 0
    
    def _estimate_page_number(self, content: str, pages_info: List[Dict], current_page: int) -> int:
        """Estimate page number based on content and position"""
        # Look for page indicators in content
        content_lower = content.lower()
        
        # Check for explicit page references
        page_match = re.search(r'page\s+(\d+)', content_lower)
        if page_match:
            try:
                return int(page_match.group(1))
            except ValueError:
                pass
        
        # Use current page as default
        return current_page
    
    def _extract_tables_from_markdown(self, content: str) -> List[ExtractedTable]:
        """Extract tables from markdown content"""
        tables = []
        
        if not content:
            return tables
        
        # Find markdown tables
        lines = content.split('\n')
        current_table = []
        table_index = 0
        page_number = 1
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if ('|' in line_stripped and 
                line_stripped.startswith('|') and 
                line_stripped.endswith('|') and
                line_stripped.count('|') >= 3):  # At least 3 | for a valid table row
                current_table.append(line_stripped)
            else:
                if current_table and len(current_table) >= 2:  # At least header + separator
                    # Process the completed table
                    table = self._parse_markdown_table(current_table, table_index, page_number)
                    if table:
                        tables.append(table)
                        table_index += 1
                
                current_table = []
                
                # Update page number estimate
                if 'page' in line.lower():
                    page_match = re.search(r'(\d+)', line)
                    if page_match:
                        try:
                            page_number = int(page_match.group(1))
                        except ValueError:
                            pass
        
        # Handle last table if exists
        if current_table and len(current_table) >= 2:
            table = self._parse_markdown_table(current_table, table_index, page_number)
            if table:
                tables.append(table)
        
        logger.info(f"Extracted {len(tables)} tables from markdown content")
        return tables
    
    def _parse_markdown_table(self, table_lines: List[str], table_index: int, page_number: int) -> Optional[ExtractedTable]:
        """Parse a markdown table into ExtractedTable format"""
        try:
            if len(table_lines) < 2:
                return None
            
            # Parse headers (first line)
            header_line = table_lines[0]
            headers = [cell.strip() for cell in header_line.split('|')[1:-1]]  # Remove empty first/last
            
            if not headers or all(not h for h in headers):
                return None
            
            # Skip separator line (second line) - check if it's actually a separator
            separator_line = table_lines[1] if len(table_lines) > 1 else ""
            if not re.match(r'^\s*\|[\s\-:]*\|\s*$', separator_line):
                # Not a proper separator, treat as data
                data_lines = table_lines[1:]
            else:
                data_lines = table_lines[2:] if len(table_lines) > 2 else []
            
            # Parse data rows
            data = []
            for line in data_lines:
                if '|' in line:
                    row = [cell.strip() for cell in line.split('|')[1:-1]]
                    # Ensure row has same length as headers
                    while len(row) < len(headers):
                        row.append('')
                    data.append(row[:len(headers)])  # Truncate if too long
            
            # Only return table if we have some data
            if not data:
                return None
            
            return ExtractedTable(
                page_number=page_number,
                table_index=table_index,
                headers=headers,
                data=data,
                table_type='markdown_table',
                confidence_score=0.95,  # High confidence for LlamaCloud extraction
                extraction_method='llamacloud_markdown'
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse markdown table: {e}")
            return None
    
    def _determine_document_type(self, content: str, tables: List[ExtractedTable]) -> DocumentType:
        """Determine document type based on content analysis"""
        if not content:
            return DocumentType.GENERAL
            
        content_lower = content.lower()
        
        # Educational indicators (for transcripts/marksheets like your example)
        educational_keywords = [
            'course', 'grade', 'transcript', 'semester', 'credit', 'cgpa', 'gpa',
            'subject', 'marks', 'student', 'education', 'curriculum', 'university',
            'syllabus', 'assignment', 'exam', 'quiz', 'result', 'marksheet'
        ]
        if (any(keyword in content_lower for keyword in educational_keywords) or 
            len(tables) > 0):  # Educational documents often have tables
            return DocumentType.EDUCATIONAL
        
        # Academic/Research indicators
        academic_keywords = [
            'abstract', 'introduction', 'methodology', 'results', 'conclusion',
            'references', 'bibliography', 'doi:', 'arxiv:', 'journal', 'research',
            'hypothesis', 'experiment', 'analysis', 'study', 'findings'
        ]
        if any(keyword in content_lower for keyword in academic_keywords):
            return DocumentType.ACADEMIC
        
        # Financial indicators
        financial_keywords = [
            'balance sheet', 'income statement', 'cash flow', 'revenue', 'profit',
            'financial', 'quarterly', 'annual report', 'earnings', 'assets',
            'liabilities', 'equity', 'investment', 'fiscal'
        ]
        if any(keyword in content_lower for keyword in financial_keywords):
            return DocumentType.FINANCIAL
        
        # Legal indicators
        legal_keywords = [
            'contract', 'agreement', 'terms and conditions', 'whereas',
            'party of the first part', 'legal', 'court', 'plaintiff', 'defendant',
            'shall', 'hereby', 'witnesseth', 'jurisdiction', 'clause'
        ]
        if any(keyword in content_lower for keyword in legal_keywords):
            return DocumentType.LEGAL
        
        # Technical indicators
        technical_keywords = [
            'algorithm', 'implementation', 'technical specification',
            'api', 'protocol', 'architecture', 'design document', 'system',
            'software', 'hardware', 'requirements', 'specification'
        ]
        if any(keyword in content_lower for keyword in technical_keywords):
            return DocumentType.TECHNICAL
        
        # Report indicators
        report_keywords = [
            'executive summary', 'recommendations', 'findings',
            'analysis', 'report', 'study', 'survey', 'overview',
            'summary', 'conclusion', 'assessment'
        ]
        if any(keyword in content_lower for keyword in report_keywords):
            return DocumentType.REPORT
        
        return DocumentType.GENERAL
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and health info"""
        try:
            return {
                'service': 'llamacloud',
                'status': 'healthy' if self.is_available() else 'unavailable',
                'api_key_configured': bool(self.api_key),
                'parser_initialized': self.parser is not None,
                'cache_size': len(self._cache),
                'llamaparse_available': LLAMAPARSE_AVAILABLE,
                'capabilities': [
                    'complex_layouts',
                    'table_extraction', 
                    'image_processing',
                    'multi_language',
                    'structured_output',
                    'vision_models'
                ] if self.is_available() else [],
                'supported_formats': ['pdf'] if self.is_available() else [],
                'max_file_size': '100MB' if self.is_available() else 'N/A',
                'concurrent_jobs': 4 if self.is_available() else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {
                'service': 'llamacloud',
                'status': 'error',
                'error': str(e)
            }
    
    def clear_cache(self):
        """Clear the processing cache"""
        self._cache.clear()
        logger.info("Processing cache cleared")


# Global instance
_llamacloud_processor = None

def get_llamacloud_processor() -> Optional[LlamaCloudProcessor]:
    """Get or create global LlamaCloud processor instance"""
    global _llamacloud_processor
    if _llamacloud_processor is None:
        try:
            _llamacloud_processor = LlamaCloudProcessor()
            if not _llamacloud_processor.is_available():
                logger.warning("LlamaCloud processor initialized but not available")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize LlamaCloud processor: {e}")
            return None
    
    return _llamacloud_processor if _llamacloud_processor.is_available() else None


# Utility functions for integration
def is_llamacloud_available() -> bool:
    """Check if LlamaCloud processing is available"""
    try:
        processor = get_llamacloud_processor()
        return processor is not None and processor.is_available()
    except Exception:
        return False


def should_use_llamacloud(pdf_content: bytes, filename: str) -> bool:
    """
    Determine if a document should use LlamaCloud processing
    based on complexity indicators
    """
    if not is_llamacloud_available():
        return False
    
    # Size-based decision (larger files likely more complex)
    if len(pdf_content) > 5 * 1024 * 1024:  # > 5MB
        return True
    
    # Filename-based hints
    filename_lower = filename.lower()
    complex_indicators = [
        'annual_report', 'financial', 'research', 'technical',
        'complex', 'diagram', 'chart', 'table', 'multi', 'marksheet', 'transcript'
    ]
    
    if any(indicator in filename_lower for indicator in complex_indicators):
        return True
    
    # Default: use LlamaCloud for better quality if available
    return True