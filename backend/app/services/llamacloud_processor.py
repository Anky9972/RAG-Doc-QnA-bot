# app/services/llamacloud_processor.py

import logging
import time
import hashlib
import os
import tempfile
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import wraps

# Try importing LlamaParse
LLAMAPARSE_AVAILABLE = False
LLAMAPARSE_ERROR = None
LlamaParse = None

try:
    from llama_parse import LlamaParse
    LLAMAPARSE_AVAILABLE = True
except ImportError as e:
    LLAMAPARSE_ERROR = f"llama_parse import failed: {str(e)}"
    try:
        from llama_cloud import LlamaParse
        LLAMAPARSE_AVAILABLE = True
        LLAMAPARSE_ERROR = None
    except ImportError as e2:
        LLAMAPARSE_ERROR = f"Both imports failed. llama_parse: {str(e)}, llama_cloud: {str(e2)}"

from app.core import config
from app.schemas import ProcessedDocument, DocumentType, TextChunk, ExtractedTable

logger = logging.getLogger(__name__)

# Log import status immediately
if LLAMAPARSE_AVAILABLE:
    logger.info("✓ LlamaParse package successfully imported")
else:
    logger.warning(f"✗ LlamaParse package not available: {LLAMAPARSE_ERROR}")


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
    full_text: str
    full_markdown: str
    pages_data: List[Dict[str, Any]]
    image_documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    raw_json: Dict[str, Any]


def retry_on_failure(max_retries=2, delay=1.0):
    """Decorator for retrying failed operations"""
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
    """Enhanced PDF processor using LlamaCloud/LlamaParse"""
    
    def __init__(self):
        self.api_key = None
        self.parser = None
        self._cache: Dict[str, ProcessedDocument] = {}
        self._initialized = False
        self._init_error = None
        self._init_error_code = None
        
        logger.info("Initializing LlamaCloudProcessor...")
        
        # Check package availability first
        if not LLAMAPARSE_AVAILABLE:
            self._init_error = f"LlamaParse package not available. {LLAMAPARSE_ERROR}"
            self._init_error_code = "PACKAGE_NOT_INSTALLED"
            logger.error(f"✗ {self._init_error}")
            logger.info("Install with: pip install llama-parse")
            return
        
        # Try to get API key
        try:
            self.api_key = self._get_api_key()
            logger.info(f"✓ API key found: {self.api_key[:10]}...")
        except LlamaCloudError as e:
            self._init_error = e.message
            self._init_error_code = e.error_code
            logger.error(f"✗ API key error: {self._init_error}")
            return
        except Exception as e:
            self._init_error = f"Unexpected error getting API key: {str(e)}"
            self._init_error_code = "API_KEY_ERROR"
            logger.error(f"✗ {self._init_error}")
            return
        
        # Try to initialize parser
        try:
            self._initialize_parser()
            self._initialized = True
            logger.info("✓ LlamaCloud processor initialized successfully")
        except Exception as e:
            self._init_error = f"Parser initialization failed: {str(e)}"
            self._init_error_code = "INIT_FAILED"
            logger.error(f"✗ {self._init_error}", exc_info=True)
            return
    
    def _get_api_key(self) -> str:
        """Get LlamaCloud API key from environment or config"""
        # Try environment variable first
        api_key = os.getenv('LLAMA_CLOUD_API_KEY')
        
        if api_key:
            logger.info("API key found in environment variable")
        else:
            # Try config file
            if hasattr(config, 'LLAMA_CLOUD_API_KEY'):
                api_key = config.LLAMA_CLOUD_API_KEY
                logger.info("API key found in config file")
            
        if not api_key:
            raise LlamaCloudError(
                "LLAMA_CLOUD_API_KEY not found in environment variables or config.\n"
                "Set it with: export LLAMA_CLOUD_API_KEY='llx-your-key-here'\n"
                "Get your API key from: https://cloud.llamaindex.ai/",
                error_code="MISSING_API_KEY"
            )
        
        # Validate key format
        api_key = api_key.strip()
        if not api_key.startswith('llx-'):
            logger.warning(f"API key format unusual (doesn't start with 'llx-'): {api_key[:10]}...")
            
        return api_key
    
    def _initialize_parser(self):
        """Initialize LlamaParse with optimal settings"""
        if not LLAMAPARSE_AVAILABLE:
            raise LlamaCloudError(
                "LlamaParse package not available", 
                error_code="PACKAGE_NOT_FOUND"
            )
            
        logger.info("Creating LlamaParse instance...")
        
        try:
            self.parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                num_workers=4,
                verbose=True,
                language="en"
            )
            logger.info("✓ LlamaParse instance created")
            
        except Exception as e:
            logger.error(f"✗ Failed to create LlamaParse: {e}", exc_info=True)
            raise LlamaCloudError(
                f"Parser initialization failed: {str(e)}", 
                error_code="INIT_FAILED",
                original_error=e
            )
    
    def is_available(self) -> bool:
        """Check if the processor is properly initialized"""
        available = self._initialized and self.parser is not None
        if not available:
            logger.debug(f"Processor not available. Initialized: {self._initialized}, Error: {self._init_error}")
        return available
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information"""
        return {
            "package_available": LLAMAPARSE_AVAILABLE,
            "package_error": LLAMAPARSE_ERROR,
            "initialized": self._initialized,
            "api_key_configured": self.api_key is not None,
            "api_key_prefix": self.api_key[:10] + "..." if self.api_key else None,
            "parser_created": self.parser is not None,
            "init_error": self._init_error,
            "init_error_code": self._init_error_code,
            "cache_size": len(self._cache)
        }
    
    @retry_on_failure(max_retries=2)
    def _parse_with_llamacloud(self, file_path: str) -> LlamaParseResult:
        """Parse document using LlamaCloud"""
        start_time = time.time()
        
        logger.info(f"Parsing with LlamaCloud: {file_path}")
        
        try:
            documents = self.parser.load_data(file_path)
            logger.info(f"✓ LlamaParse returned {len(documents)} document(s)")
            
        except Exception as e:
            logger.error(f"✗ LlamaParse API call failed: {e}", exc_info=True)
            raise LlamaCloudError(
                f"LlamaParse API failed: {str(e)}", 
                error_code="PARSE_FAILED",
                original_error=e
            )
        
        # Process documents
        full_text = ""
        full_markdown = ""
        pages_data = []
        processed_images = []
        
        for doc_idx, doc in enumerate(documents):
            doc_text = doc.text if hasattr(doc, 'text') else str(doc)
            full_text += doc_text + "\n\n"
            
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'markdown' in doc.metadata:
                    full_markdown += doc.metadata['markdown'] + "\n\n"
                else:
                    full_markdown += doc_text + "\n\n"
                
                page_num = doc.metadata.get('page_number', doc.metadata.get('page', doc_idx + 1))
                
                page_data = {
                    'page_number': page_num,
                    'text': doc_text,
                    'markdown': doc.metadata.get('markdown', doc_text),
                    'has_images': 'images' in doc.metadata,
                    'has_tables': '|' in doc_text and doc_text.count('|') > 5,
                    'word_count': len(doc_text.split()),
                    'metadata': doc.metadata
                }
                pages_data.append(page_data)
            else:
                full_markdown += doc_text + "\n\n"
        
        if not pages_data and full_text:
            pages_data = [{
                'page_number': 1,
                'text': full_text,
                'markdown': full_markdown,
                'has_images': False,
                'has_tables': '|' in full_text and full_text.count('|') > 5,
                'word_count': len(full_text.split()),
                'metadata': {}
            }]
        
        metadata = {
            'parsing_time': time.time() - start_time,
            'num_pages': len(pages_data),
            'num_documents': len(documents),
            'total_words': sum(p['word_count'] for p in pages_data),
            'parser_version': 'llamacloud',
            'language': 'en'
        }
        
        logger.info(f"✓ Parsed: {len(pages_data)} pages, {metadata['total_words']} words")
        
        return LlamaParseResult(
            full_text=full_text,
            full_markdown=full_markdown,
            pages_data=pages_data,
            image_documents=processed_images,
            metadata=metadata,
            processing_time=time.time() - start_time,
            raw_json={'full_text': full_text, 'pages': pages_data}
        )
    
    def _create_intelligent_chunks(
        self, 
        markdown_content: str, 
        pages_data: List[Dict[str, Any]],
        max_chunk_size: int = 1500
    ) -> List[TextChunk]:
        """Create intelligent text chunks"""
        chunks = []
        chunk_id = 0
        
        if not markdown_content:
            logger.warning("No markdown content to chunk")
            return chunks
        
        sections = self._smart_content_split(markdown_content, max_chunk_size)
        logger.info(f"Split into {len(sections)} intelligent sections")
        
        for section in sections:
            if not section.strip():
                continue
            
            page_number = self._estimate_page_for_section(section, pages_data)
            content_type = self._classify_content_type(section)
            
            chunk = TextChunk(
                id=chunk_id,
                text=section.strip(),
                page_number=page_number,
                chunk_type=content_type,
                metadata={
                    'extraction_method': 'llamacloud',
                    'content_type': content_type,
                    'word_count': len(section.split()),
                    'quality_score': 0.95
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
        
        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks
    
    def _smart_content_split(self, content: str, max_chunk_size: int = 1500) -> List[str]:
        """Split content intelligently by structure"""
        sections = []
        lines = content.split('\n')
        current_section = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            is_header = line.strip().startswith('#')
            
            should_split = (
                (is_header and current_size > 300) or
                (current_size + line_size > max_chunk_size and current_size > 500)
            )
            
            if should_split and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line] if line.strip() else []
                current_size = line_size
            else:
                current_section.append(line)
                current_size += line_size
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return [s.strip() for s in sections if s.strip()]
    
    def _estimate_page_for_section(self, section: str, pages_data: List[Dict[str, Any]]) -> int:
        """Estimate page number for section"""
        if not pages_data:
            return 1
        
        section_words = set(section.lower().split()[:20])
        best_match_page = 1
        best_match_score = 0
        
        for page in pages_data:
            page_text = page.get('text', '').lower()
            page_words = set(page_text.split())
            overlap = len(section_words & page_words)
            
            if overlap > best_match_score:
                best_match_score = overlap
                best_match_page = page['page_number']
        
        return best_match_page
    
    def _classify_content_type(self, content: str) -> str:
        """Classify content type"""
        if '|' in content and content.count('|') > 5:
            return 'table'
        elif content.strip().startswith('#'):
            return 'header'
        elif len(content.split()) > 100:
            return 'paragraph'
        return 'text'
    
    def process_pdf_comprehensive(
        self, 
        pdf_content: bytes, 
        filename: str = "document.pdf"
    ) -> ProcessedDocument:
        """Process PDF with LlamaCloud"""
        
        if not self.is_available():
            error_msg = f"LlamaCloud processor not available: {self._init_error}"
            logger.error(error_msg)
            raise LlamaCloudError(error_msg, error_code=self._init_error_code)
        
        cache_key = hashlib.sha256(pdf_content).hexdigest()
        if cache_key in self._cache:
            logger.info(f"Returning cached result for {filename}")
            return self._cache[cache_key]
        
        tmp_file_path = None
        
        try:
            logger.info(f"Processing {filename} ({len(pdf_content)} bytes) with LlamaCloud")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file_path = tmp_file.name
            
            llama_result = self._parse_with_llamacloud(tmp_file_path)
            text_chunks = self._create_intelligent_chunks(
                llama_result.full_markdown,
                llama_result.pages_data
            )
            
            tables = []  # Implement table extraction if needed
            
            document_metadata = {
                'filename': filename,
                'file_size': len(pdf_content),
                'pdf_hash': cache_key,
                'page_count': llama_result.metadata['num_pages'],
                'processing_method': 'llamacloud',
                'chunk_count': len(text_chunks)
            }
            
            processed_doc = ProcessedDocument(
                text_chunks=text_chunks,
                tables=tables,
                document_metadata=document_metadata,
                document_type=DocumentType.GENERAL,
                processing_method='llamacloud'
            )
            
            self._cache[cache_key] = processed_doc
            logger.info(f"✓ Successfully processed {filename}: {len(text_chunks)} chunks")
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"✗ Processing failed: {e}", exc_info=True)
            raise
            
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'service': 'llamacloud',
            'status': 'healthy' if self.is_available() else 'unavailable',
            'error': self._init_error,
            'cache_size': len(self._cache),
            'capabilities': [
                'intelligent_chunking',
                'markdown_output',
                'table_extraction'
            ] if self.is_available() else []
        }


# Global instance
_llamacloud_processor = None

def get_llamacloud_processor() -> Optional[LlamaCloudProcessor]:
    """Get global processor instance"""
    global _llamacloud_processor
    if _llamacloud_processor is None:
        _llamacloud_processor = LlamaCloudProcessor()
    
    return _llamacloud_processor if _llamacloud_processor.is_available() else None


def is_llamacloud_available() -> bool:
    """Check if LlamaCloud is available"""
    try:
        processor = get_llamacloud_processor()
        is_avail = processor is not None and processor.is_available()
        logger.debug(f"LlamaCloud available: {is_avail}")
        return is_avail
    except Exception as e:
        logger.error(f"Error checking availability: {e}")
        return False


def should_use_llamacloud(pdf_content: bytes, filename: str) -> bool:
    """Determine if should use LlamaCloud"""
    if not is_llamacloud_available():
        logger.info("LlamaCloud not available, using fallback")
        return False
    
    # Use for ALL files when available (remove size restriction)
    logger.info(f"LlamaCloud available, will use for {filename}")
    return True