# services/hybrid_search.py
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateQueryException
from app.core import config
from app.services import vector_store
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class QueryExpander:
    def __init__(self):
        self.synonyms = {
            'analyze': ['examine', 'study', 'evaluate', 'assess'],
            'explain': ['describe', 'clarify', 'elaborate', 'detail'],
            'compare': ['contrast', 'differentiate', 'relate', 'distinguish'],
            'summarize': ['outline', 'overview', 'recap', 'synopsis'],
            'important': ['significant', 'crucial', 'key', 'vital', 'essential'],
            'method': ['approach', 'technique', 'procedure', 'process'],
            'result': ['outcome', 'finding', 'conclusion', 'effect']
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and variations"""
        expanded_queries = [query]
        words = query.lower().split()
        
        for word in words:
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    expanded_query = query.lower().replace(word, synonym)
                    expanded_queries.append(expanded_query)
        
        return list(set(expanded_queries))  # Remove duplicates

class KeywordSearchEngine:
    def __init__(self):
        self.vectorizer = None
        self.doc_vectors = None
        self.documents = []
        
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for keyword search using TF-IDF"""
        self.documents = documents
        texts = [doc.get('text', '') for doc in documents]
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000,
            lowercase=True
        )
        
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        logger.info(f"Indexed {len(documents)} documents for keyword search")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform keyword search and return document indices with scores"""
        if not self.vectorizer or self.doc_vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]
        
        return results

class HybridSearchEngine:
    def __init__(self):
        self.embedding_model = None
        self.rerank_model = None
        self.keyword_engine = KeywordSearchEngine()
        self.query_expander = QueryExpander()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding and re-ranking models"""
        try:
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            logger.info(f"Loaded embedding model: {config.EMBEDDING_MODEL_NAME}")
            
            self.rerank_model = CrossEncoder(config.RERANK_MODEL_NAME)
            logger.info(f"Loaded re-ranking model: {config.RERANK_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Error initializing search models: {e}")
    
    async def vector_search(
        self, 
        client: weaviate.WeaviateClient, 
        query: str, 
        document_id: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        if not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Weaviate
            collection = client.collections.get(config.WEAVIATE_CLASS_NAME)
            response = collection.query.near_vector(
                near_vector=query_embedding,
                filters=wvc.query.Filter.by_property("pdf_id").equal(document_id),
                limit=top_k,
                return_metadata=wvc.query.MetadataQuery(distance=True),
                return_properties=["text", "page_number", "pdf_id"]
            )
            
            results = []
            for obj in response.objects:
                distance = obj.metadata.distance if obj.metadata else 1.0
                results.append({
                    "text": obj.properties.get("text", ""),
                    "page_number": obj.properties.get("page_number", 0),
                    "pdf_id": obj.properties.get("pdf_id", ""),
                    "vector_score": 1 - distance,  # Convert distance to similarity score
                    "search_type": "vector",
                    "_additional": {
                        "distance": distance,
                        "id": str(obj.uuid) if obj.uuid else None
                    }
                })
            
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def keyword_search_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        query: str, 
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform keyword search on document chunks"""
        if not chunks:
            return []
        
        # Index chunks for keyword search
        self.keyword_engine.index_documents(chunks)
        
        # Expand query for better recall
        expanded_queries = self.query_expander.expand_query(query)
        
        # Combine results from all expanded queries
        all_results = defaultdict(float)
        for expanded_query in expanded_queries:
            keyword_results = self.keyword_engine.search(expanded_query, top_k)
            for idx, score in keyword_results:
                all_results[idx] = max(all_results[idx], score)
        
        # Convert to final format
        results = []
        for idx, score in sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if idx < len(chunks):
                chunk = chunks[idx].copy()
                chunk.update({
                    "keyword_score": score,
                    "search_type": "keyword"
                })
                results.append(chunk)
        
        return results
    
    def combine_search_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Combine vector and keyword search results"""
        
        # Create a map for combining results
        combined_results = {}
        
        # Add vector results
        for result in vector_results:
            text_key = result.get('text', '')[:100]  # Use first 100 chars as key
            if text_key:
                combined_results[text_key] = result.copy()
                combined_results[text_key]['combined_score'] = result.get('vector_score', 0) * vector_weight
        
        # Add keyword results
        for result in keyword_results:
            text_key = result.get('text', '')[:100]
            if text_key in combined_results:
                # Combine scores for existing results
                combined_results[text_key]['combined_score'] += result.get('keyword_score', 0) * keyword_weight
                combined_results[text_key]['search_type'] = 'hybrid'
            else:
                # Add new keyword-only results
                combined_results[text_key] = result.copy()
                combined_results[text_key]['combined_score'] = result.get('keyword_score', 0) * keyword_weight
        
        # Sort by combined score
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x.get('combined_score', 0),
            reverse=True
        )
        
        return final_results
    
    async def rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Re-rank search results using cross-encoder"""
        if not self.rerank_model or not results:
            return results[:top_k]
        
        try:
            # Prepare query-document pairs for re-ranking
            pairs = []
            for result in results:
                pairs.append([query, result.get('text', '')])
            
            # Get re-ranking scores
            rerank_scores = self.rerank_model.predict(pairs)
            
            # Add rerank scores to results
            for i, result in enumerate(results):
                result['rerank_score'] = float(rerank_scores[i])
            
            # Sort by rerank score
            reranked_results = sorted(
                results,
                key=lambda x: x.get('rerank_score', 0),
                reverse=True
            )
            
            return reranked_results[:top_k]
        
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results[:top_k]
    
    async def hybrid_search(
        self,
        client: weaviate.WeaviateClient,
        query: str,
        document_id: str,
        top_k: int = 5,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search"""
        
        config_hybrid = config.HYBRID_SEARCH_CONFIG
        
        # Step 1: Vector search
        vector_results = await self.vector_search(
            client, query, document_id, 
            top_k=config_hybrid.get('rerank_top_k', 20)
        )
        
        # Step 2: Get all chunks for keyword search
        all_chunks = await self._get_document_chunks(client, document_id)
        
        # Step 3: Keyword search
        keyword_results = self.keyword_search_chunks(
            all_chunks, query,
            top_k=config_hybrid.get('rerank_top_k', 20)
        )
        
        # Step 4: Combine results
        combined_results = self.combine_search_results(
            vector_results,
            keyword_results,
            vector_weight=config_hybrid.get('vector_weight', 0.7),
            keyword_weight=config_hybrid.get('keyword_weight', 0.3)
        )
        
        # Step 5: Re-rank if enabled
        if use_reranking and self.rerank_model:
            final_results = await self.rerank_results(
                query, combined_results, top_k
            )
        else:
            final_results = combined_results[:top_k]
        
        # Add search metadata
        for result in final_results:
            result['search_metadata'] = {
                'hybrid_search': True,
                'reranked': use_reranking,
                'vector_weight': config_hybrid.get('vector_weight', 0.7),
                'keyword_weight': config_hybrid.get('keyword_weight', 0.3)
            }
        
        logger.info(f"Hybrid search returned {len(final_results)} results for query: {query[:50]}...")
        return final_results
    
    async def _get_document_chunks(
        self, 
        client: weaviate.WeaviateClient, 
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        try:
            collection = client.collections.get(config.WEAVIATE_CLASS_NAME)
            response = collection.query.fetch_objects(
                filters=wvc.query.Filter.by_property("pdf_id").equal(document_id),
                limit=1000,  # Adjust based on expected document size
                return_properties=["text", "page_number", "pdf_id"]
            )
            
            chunks = []
            for obj in response.objects:
                chunks.append({
                    "text": obj.properties.get("text", ""),
                    "page_number": obj.properties.get("page_number", 0),
                    "pdf_id": obj.properties.get("pdf_id", "")
                })
            
            return chunks
        except Exception as e:
            logger.error(f"Error fetching document chunks: {e}")
            return []

# Initialize global hybrid search engine
hybrid_search_engine = HybridSearchEngine()