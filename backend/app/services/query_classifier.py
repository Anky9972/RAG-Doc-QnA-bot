# services/query_classifier.py - NEW FILE

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from sqlalchemy.orm import Session
import spacy
from app.database.models import DocumentTable, TableEntity, Document

logger = logging.getLogger(__name__)

class QueryType(Enum):
    SEMANTIC = "semantic"       # Current capability - general Q&A
    STRUCTURED = "structured"   # Table/data queries - NEW
    ANALYTICAL = "analytical"   # Analysis across data - NEW
    HYBRID = "hybrid"          # Combination of structured + semantic

class EntityType(Enum):
    SUBJECT = "subject"
    TEACHER = "teacher" 
    INSTRUCTOR = "instructor"
    PROFESSOR = "professor"
    CREDITS = "credits"
    HOURS = "hours"
    SEMESTER = "semester"
    DEPARTMENT = "department"
    COURSE_CODE = "course_code"
    SCHEDULE = "schedule"

@dataclass
class QueryClassification:
    query_type: QueryType
    confidence: float
    entities: List[Dict[str, Any]]
    intent: str
    suggested_response_format: str

@dataclass  
class StructuredQueryResult:
    data: List[Dict[str, Any]]
    total_matches: int
    query_sql: Optional[str]
    metadata: Dict[str, Any]

class QueryClassifier:
    def __init__(self):
        # Patterns for different query types
        self.structured_patterns = {
            'catalog_queries': [
                r'(which|what)\s+(teacher|instructor|professor|faculty)\s+(teaches?|is\s+teaching)',
                r'(who|which)\s+(teaches?|is\s+teaching)\s+(.+)',
                r'(how\s+many|what)\s+(credits?|hours?)',
                r'(list|show|find)\s+(all|the)?\s*(subjects?|courses?)',
                r'subjects?\s+(taught\s+by|in\s+semester)',
                r'(credits?|hours?)\s+(for|of)\s+(.+)',
                r'semester\s+\d+\s+(subjects?|courses?)',
                r'(course\s+code|subject\s+code)',
                r'teachers?\s+(for|in)\s+(semester|department)',
            ],
            'analytical_queries': [
                r'(compare|contrast)\s+(.+)\s+(with|and|vs)',
                r'(total|sum|average|mean)\s+(credits?|hours?)',
                r'(most|least)\s+(credits?|hours?|popular)',
                r'(statistics?|summary)\s+(of|for|about)',
                r'(distribution|breakdown)\s+(of|by)',
                r'(trend|pattern)\s+(in|of|across)'
            ],
            'hybrid_queries': [
                r'(explain|describe)\s+(.+)\s+(taught\s+by|in\s+semester)',
                r'(what\s+is|tell\s+me\s+about)\s+(.+)\s+(subject|course)',
                r'(details?|information)\s+(about|on)\s+(.+)\s+(teacher|course)'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            EntityType.SUBJECT: [
                r'(subject|course)\s*:?\s*([A-Z]+\s*\d+|[A-Za-z\s]+)',
                r'([A-Z]{2,4}\s*\d{3,4})',  # Course codes like CS101, MATH201
            ],
            EntityType.TEACHER: [
                r'(teacher|instructor|professor|faculty)\s*:?\s*([A-Za-z\s\.]+)',
                r'(taught\s+by|teacher\s+is|instructor\s+is)\s+([A-Za-z\s\.]+)',
                r'(Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s+([A-Za-z\s]+)'
            ],
            EntityType.CREDITS: [
                r'(\d+)\s*(credits?|credit\s+hours?)',
                r'(credits?|credit\s+hours?)\s*:?\s*(\d+)'
            ],
            EntityType.HOURS: [
                r'(\d+)\s*(hours?|hrs?)',
                r'(hours?|hrs?)\s*:?\s*(\d+)'
            ],
            EntityType.SEMESTER: [
                r'(semester\s*\d+|sem\s*\d+)',
                r'(\d+)(st|nd|rd|th)?\s*semester'
            ]
        }
        
        # Try to load spaCy model for advanced NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def classify_query(self, query: str) -> QueryClassification:
        """Main function to classify queries"""
        query_lower = query.lower().strip()
        
        # Check for structured patterns
        structured_score = self._calculate_pattern_score(query_lower, 'catalog_queries')
        analytical_score = self._calculate_pattern_score(query_lower, 'analytical_queries') 
        hybrid_score = self._calculate_pattern_score(query_lower, 'hybrid_queries')
        
        # Determine query type
        max_score = max(structured_score, analytical_score, hybrid_score)
        
        if max_score == 0:
            query_type = QueryType.SEMANTIC
            confidence = 0.8  # Default confidence for semantic
        elif structured_score == max_score:
            query_type = QueryType.STRUCTURED
            confidence = min(0.9, 0.6 + structured_score * 0.1)
        elif analytical_score == max_score:
            query_type = QueryType.ANALYTICAL
            confidence = min(0.9, 0.6 + analytical_score * 0.1)
        else:
            query_type = QueryType.HYBRID
            confidence = min(0.9, 0.6 + hybrid_score * 0.1)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine intent
        intent = self._determine_intent(query_lower, query_type)
        
        # Suggest response format
        response_format = self._suggest_response_format(query_type, entities)
        
        return QueryClassification(
            query_type=query_type,
            confidence=confidence,
            entities=entities,
            intent=intent,
            suggested_response_format=response_format
        )
    
    def _calculate_pattern_score(self, query: str, pattern_category: str) -> float:
        """Calculate how well query matches patterns in a category"""
        if pattern_category not in self.structured_patterns:
            return 0.0
        
        patterns = self.structured_patterns[pattern_category]
        matches = 0
        
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matches += 1
        
        return matches / len(patterns) if patterns else 0.0
    
    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query using patterns and NLP"""
        entities = []
        
        # Pattern-based extraction
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    # Extract the relevant group (usually the last one)
                    groups = match.groups()
                    if groups:
                        value = groups[-1].strip()
                        if value and len(value) > 1:
                            entities.append({
                                'type': entity_type.value,
                                'value': value,
                                'confidence': 0.8,
                                'source': 'pattern'
                            })
        
        # SpaCy-based extraction if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                # Map spaCy entities to our types
                entity_type = self._map_spacy_entity(ent.label_)
                if entity_type:
                    entities.append({
                        'type': entity_type,
                        'value': ent.text,
                        'confidence': 0.7,
                        'source': 'spacy'
                    })
        
        # Deduplicate entities
        return self._deduplicate_entities(entities)
    
    def _map_spacy_entity(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            'PERSON': EntityType.TEACHER.value,
            'ORG': EntityType.DEPARTMENT.value,
            'CARDINAL': EntityType.CREDITS.value,  # Numbers might be credits/hours
        }
        return mapping.get(spacy_label)
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities, keeping highest confidence"""
        seen = {}
        for entity in entities:
            key = (entity['type'], entity['value'].lower())
            if key not in seen or entity['confidence'] > seen[key]['confidence']:
                seen[key] = entity
        return list(seen.values())
    
    def _determine_intent(self, query: str, query_type: QueryType) -> str:
        """Determine the specific intent of the query"""
        
        intent_patterns = {
            'find_teacher': [r'(who|which)\s+(teaches?|is\s+teaching)', r'teacher\s+(for|of)'],
            'find_subjects': [r'(list|show|find)\s+.*subjects?', r'subjects?\s+(in|for)'],
            'get_credits': [r'(how\s+many|what)\s+credits?', r'credits?\s+(for|of)'],
            'get_hours': [r'(how\s+many|what)\s+hours?', r'hours?\s+(for|of)'],
            'list_by_semester': [r'semester\s+\d+', r'(subjects?|courses?)\s+.*semester'],
            'compare': [r'(compare|contrast)', r'difference\s+between'],
            'analyze': [r'(analyze|analysis)', r'(statistics?|summary)'],
            'explain': [r'(explain|describe|what\s+is)', r'tell\s+me\s+about']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
                return intent
        
        return 'general_query'
    
    def _suggest_response_format(self, query_type: QueryType, entities: List[Dict[str, Any]]) -> str:
        """Suggest the best response format based on query type and entities"""
        
        if query_type == QueryType.STRUCTURED:
            if any(e['type'] in ['subject', 'teacher'] for e in entities):
                return 'structured_table'
            else:
                return 'structured_list'
        elif query_type == QueryType.ANALYTICAL:
            return 'summary_with_statistics'
        elif query_type == QueryType.HYBRID:
            return 'narrative_with_data'
        else:
            return 'natural_language'


class StructuredQueryProcessor:
    """Process structured queries against table data"""
    
    def __init__(self):
        self.query_classifier = QueryClassifier()
    
    async def process_structured_query(
        self,
        query: str,
        document_id: str,
        db: Session
    ) -> StructuredQueryResult:
        """Process a structured query and return formatted results"""
        
        # Classify the query first
        classification = self.query_classifier.classify_query(query)
        
        if classification.query_type not in [QueryType.STRUCTURED, QueryType.ANALYTICAL]:
            raise ValueError("Query is not structured - use semantic search instead")
        
        # Get document tables
        tables = db.query(DocumentTable).filter(
            DocumentTable.document_id == document_id,
            DocumentTable.processed == True
        ).all()
        
        if not tables:
            return StructuredQueryResult(
                data=[],
                total_matches=0,
                query_sql=None,
                metadata={'error': 'No structured data available for this document'}
            )
        
        # Process based on intent
        if classification.intent == 'find_teacher':
            return await self._process_teacher_query(query, tables, classification.entities, db)
        elif classification.intent == 'find_subjects':
            return await self._process_subject_query(query, tables, classification.entities, db)
        elif classification.intent == 'get_credits':
            return await self._process_credits_query(query, tables, classification.entities, db)
        elif classification.intent == 'list_by_semester':
            return await self._process_semester_query(query, tables, classification.entities, db)
        else:
            return await self._process_general_structured_query(query, tables, classification, db)
    
    async def _process_teacher_query(
        self,
        query: str,
        tables: List[DocumentTable],
        entities: List[Dict[str, Any]],
        db: Session
    ) -> StructuredQueryResult:
        """Process queries about teachers/instructors"""
        
        results = []
        
        # Look for teacher entities in the tables
        for table in tables:
            if table.table_type != 'course_catalog':
                continue
            
            # Search through table data
            for row_idx, row_data in enumerate(table.table_data):
                row_text = ' '.join([str(cell) for cell in row_data]).lower()
                
                # Check if this row contains teacher information
                teacher_patterns = [
                    r'(dr\.|prof\.|mr\.|ms\.|mrs\.)\s*[a-zA-Z\s]+',
                    r'[a-zA-Z]+\s*,\s*[a-zA-Z]+',  # Last, First format
                    r'instructor\s*:?\s*[a-zA-Z\s]+'
                ]
                
                teacher_match = None
                for pattern in teacher_patterns:
                    match = re.search(pattern, row_text, re.IGNORECASE)
                    if match:
                        teacher_match = match.group().strip()
                        break
                
                if teacher_match:
                    # Extract other information from the same row
                    row_dict = {
                        'teacher': teacher_match,
                        'table_id': table.id,
                        'page_number': table.page_number,
                        'row_index': row_idx,
                        'raw_data': row_data
                    }
                    
                    # Try to identify subject, credits, hours from the row
                    subject_match = self._extract_subject_from_row(row_data)
                    if subject_match:
                        row_dict['subject'] = subject_match
                    
                    credits_match = self._extract_credits_from_row(row_data)
                    if credits_match:
                        row_dict['credits'] = credits_match
                    
                    hours_match = self._extract_hours_from_row(row_data)
                    if hours_match:
                        row_dict['hours'] = hours_match
                    
                    results.append(row_dict)
        
        # Filter results based on query entities if any specific subject/teacher mentioned
        subject_entities = [e for e in entities if e['type'] == 'subject']
        teacher_entities = [e for e in entities if e['type'] in ['teacher', 'instructor', 'professor']]
        
        if subject_entities:
            subject_terms = [e['value'].lower() for e in subject_entities]
            results = [r for r in results if any(term in r.get('subject', '').lower() for term in subject_terms)]
        
        if teacher_entities:
            teacher_terms = [e['value'].lower() for e in teacher_entities]
            results = [r for r in results if any(term in r.get('teacher', '').lower() for term in teacher_terms)]
        
        return StructuredQueryResult(
            data=results,
            total_matches=len(results),
            query_sql=None,  # We're not using SQL, but processing table data directly
            metadata={
                'query_type': 'teacher_search',
                'entities_found': entities,
                'tables_processed': len(tables)
            }
        )
    
    async def _process_subject_query(
        self,
        query: str,
        tables: List[DocumentTable],
        entities: List[Dict[str, Any]],
        db: Session
    ) -> StructuredQueryResult:
        """Process queries about subjects/courses"""
        
        results = []
        
        for table in tables:
            if table.table_type != 'course_catalog':
                continue
            
            for row_idx, row_data in enumerate(table.table_data):
                # Look for subject/course information
                subject_match = self._extract_subject_from_row(row_data)
                
                if subject_match:
                    row_dict = {
                        'subject': subject_match,
                        'table_id': table.id,
                        'page_number': table.page_number,
                        'row_index': row_idx,
                        'raw_data': row_data
                    }
                    
                    # Extract related information
                    teacher_match = self._extract_teacher_from_row(row_data)
                    if teacher_match:
                        row_dict['teacher'] = teacher_match
                    
                    credits_match = self._extract_credits_from_row(row_data)
                    if credits_match:
                        row_dict['credits'] = credits_match
                    
                    hours_match = self._extract_hours_from_row(row_data)
                    if hours_match:
                        row_dict['hours'] = hours_match
                    
                    results.append(row_dict)
        
        # Filter by semester if mentioned
        semester_entities = [e for e in entities if e['type'] == 'semester']
        if semester_entities:
            # This would require more sophisticated semester detection in the data
            pass
        
        return StructuredQueryResult(
            data=results,
            total_matches=len(results),
            query_sql=None,
            metadata={
                'query_type': 'subject_search',
                'entities_found': entities,
                'tables_processed': len(tables)
            }
        )
    
    async def _process_credits_query(
        self,
        query: str,
        tables: List[DocumentTable],
        entities: List[Dict[str, Any]],
        db: Session
    ) -> StructuredQueryResult:
        """Process queries about credits"""
        
        results = []
        
        for table in tables:
            for row_idx, row_data in enumerate(table.table_data):
                credits_match = self._extract_credits_from_row(row_data)
                
                if credits_match:
                    row_dict = {
                        'credits': credits_match,
                        'table_id': table.id,
                        'page_number': table.page_number,
                        'row_index': row_idx
                    }
                    
                    # Try to get subject for context
                    subject_match = self._extract_subject_from_row(row_data)
                    if subject_match:
                        row_dict['subject'] = subject_match
                    
                    results.append(row_dict)
        
        return StructuredQueryResult(
            data=results,
            total_matches=len(results),
            query_sql=None,
            metadata={'query_type': 'credits_search'}
        )
    
    async def _process_semester_query(
        self,
        query: str,
        tables: List[DocumentTable],
        entities: List[Dict[str, Any]],
        db: Session
    ) -> StructuredQueryResult:
        """Process queries about specific semesters"""
        
        semester_entities = [e for e in entities if e['type'] == 'semester']
        if not semester_entities:
            return StructuredQueryResult(data=[], total_matches=0, query_sql=None, 
                                       metadata={'error': 'No semester specified'})
        
        target_semester = semester_entities[0]['value']
        results = []
        
        for table in tables:
            # Check if table contains semester information
            table_text = str(table.table_data).lower()
            
            if target_semester.lower() in table_text:
                # Process the entire table for this semester
                for row_idx, row_data in enumerate(table.table_data):
                    row_text = ' '.join([str(cell) for cell in row_data]).lower()
                    
                    if target_semester.lower() in row_text:
                        row_dict = {
                            'semester': target_semester,
                            'row_data': row_data,
                            'table_id': table.id,
                            'page_number': table.page_number
                        }
                        
                        # Extract other relevant info
                        subject_match = self._extract_subject_from_row(row_data)
                        if subject_match:
                            row_dict['subject'] = subject_match
                        
                        teacher_match = self._extract_teacher_from_row(row_data)
                        if teacher_match:
                            row_dict['teacher'] = teacher_match
                        
                        results.append(row_dict)
        
        return StructuredQueryResult(
            data=results,
            total_matches=len(results),
            query_sql=None,
            metadata={'query_type': 'semester_search', 'semester': target_semester}
        )
    
    async def _process_general_structured_query(
        self,
        query: str,
        tables: List[DocumentTable],
        classification: QueryClassification,
        db: Session
    ) -> StructuredQueryResult:
        """Fallback for general structured queries"""
        
        results = []
        query_terms = query.lower().split()
        
        for table in tables:
            for row_idx, row_data in enumerate(table.table_data):
                row_text = ' '.join([str(cell) for cell in row_data]).lower()
                
                # Simple relevance scoring
                relevance_score = sum(1 for term in query_terms if term in row_text)
                
                if relevance_score > 0:
                    results.append({
                        'relevance_score': relevance_score,
                        'row_data': row_data,
                        'table_id': table.id,
                        'page_number': table.page_number,
                        'row_index': row_idx
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return StructuredQueryResult(
            data=results[:10],  # Top 10 results
            total_matches=len(results),
            query_sql=None,
            metadata={'query_type': 'general_structured'}
        )
    
    # Helper methods for extracting information from table rows
    def _extract_subject_from_row(self, row_data: List[str]) -> Optional[str]:
        """Extract subject/course name from a table row"""
        patterns = [
            r'[A-Z]{2,4}\s*\d{3,4}',  # Course codes like CS101
            r'(mathematics|math|science|physics|chemistry|biology|english|history)',
        ]
        
        row_text = ' '.join([str(cell) for cell in row_data])
        
        for pattern in patterns:
            match = re.search(pattern, row_text, re.IGNORECASE)
            if match:
                return match.group().strip()
        
        return None
    
    def _extract_teacher_from_row(self, row_data: List[str]) -> Optional[str]:
        """Extract teacher name from a table row"""
        patterns = [
            r'(dr\.|prof\.|mr\.|ms\.|mrs\.)\s*[a-zA-Z\s]+',
            r'[a-zA-Z]+\s*,\s*[a-zA-Z]+',  # Last, First
        ]
        
        row_text = ' '.join([str(cell) for cell in row_data])
        
        for pattern in patterns:
            match = re.search(pattern, row_text, re.IGNORECASE)
            if match:
                return match.group().strip()
        
        return None
    
    def _extract_credits_from_row(self, row_data: List[str]) -> Optional[str]:
        """Extract credit information from a table row"""
        row_text = ' '.join([str(cell) for cell in row_data])
        
        patterns = [
            r'(\d+)\s*(credits?|cr)',
            r'(credits?|cr)\s*:?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, row_text, re.IGNORECASE)
            if match:
                # Extract the number
                numbers = re.findall(r'\d+', match.group())
                if numbers:
                    return numbers[0]
        
        return None
    
    def _extract_hours_from_row(self, row_data: List[str]) -> Optional[str]:
        """Extract hours information from a table row"""
        row_text = ' '.join([str(cell) for cell in row_data])
        
        patterns = [
            r'(\d+)\s*(hours?|hrs?)',
            r'(hours?|hrs?)\s*:?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, row_text, re.IGNORECASE)
            if match:
                numbers = re.findall(r'\d+', match.group())
                if numbers:
                    return numbers[0]
        
        return None


# Global instances
query_classifier = QueryClassifier()
structured_query_processor = StructuredQueryProcessor()