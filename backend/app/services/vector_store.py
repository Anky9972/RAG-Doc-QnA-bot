# vector_store.py
import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateQueryException, WeaviateStartUpError
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union
from app.core import config
from app.schemas import TextChunk  # Import the TextChunk dataclass
import logging

logger = logging.getLogger(__name__)

embedding_model = None
weaviate_client = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return embedding_model

def get_weaviate_client():
    global weaviate_client
    if weaviate_client is None:
        logger.info(f"Initializing Weaviate client for URL: {config.WEAVIATE_URL}")
        try:
            host = "localhost"
            port = 8080
            grpc_port = 50051

            if config.WEAVIATE_URL:
                from urllib.parse import urlparse
                parsed_url = urlparse(config.WEAVIATE_URL)
                host = parsed_url.hostname or "localhost"
                port = parsed_url.port or 8080
            
            logger.info(f"Attempting to connect to Weaviate at host={host}, port={port}, grpc_port={grpc_port}")
            
            weaviate_client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port
            )
            
            if not weaviate_client.is_ready():
                 raise WeaviateStartUpError("Weaviate is not ready after connection attempt.")
            logger.info("Weaviate client initialized and ready.")
            create_schema_if_not_exists(weaviate_client)
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}", exc_info=True)
            if weaviate_client:
                try:
                    weaviate_client.close()
                except Exception as close_e:
                    logger.error(f"Error closing Weaviate client during cleanup: {close_e}")
            weaviate_client = None 
            raise
    return weaviate_client

def create_schema_if_not_exists(client: weaviate.WeaviateClient):
    class_name = config.WEAVIATE_CLASS_NAME
    if not client.collections.exists(class_name):
        logger.info(f"Schema for class '{class_name}' does not exist. Creating...")
        try:
            client.collections.create(
                name=class_name,
                description="Stores PDF chunks and their embeddings",
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT, description="The actual text content of the chunk"),
                    wvc.config.Property(
                        name="pdf_id", 
                        data_type=wvc.config.DataType.TEXT, 
                        description="Identifier for the PDF document",
                        tokenization=wvc.config.Tokenization.KEYWORD 
                    ),
                    wvc.config.Property(name="page_number", data_type=wvc.config.DataType.INT, description="Page number from the original PDF"),
                ]
            )
            logger.info(f"Schema for class '{class_name}' created.")
        except Exception as e:
            logger.error(f"Error creating schema for class '{class_name}': {e}", exc_info=True)
            raise
    else:
        logger.info(f"Schema for class '{class_name}' already exists.")

def embed_chunks(chunks_data: Union[List[Dict[str, any]], List[TextChunk]]) -> List[List[float]]:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks_data: Either a list of dictionaries with 'text' key or a list of TextChunk objects
    
    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()
    
    # Handle both dictionary and TextChunk object formats
    texts_to_embed = []
    for chunk in chunks_data:
        if isinstance(chunk, TextChunk):
            texts_to_embed.append(chunk.text)
        elif isinstance(chunk, dict):
            texts_to_embed.append(chunk['text'])
        else:
            logger.error(f"Unexpected chunk type: {type(chunk)}")
            continue
    
    if not texts_to_embed:
        logger.warning("No text found to embed")
        return []
    
    logger.info(f"Embedding {len(texts_to_embed)} text chunks")
    embeddings = model.encode(texts_to_embed, convert_to_tensor=False).tolist()
    return embeddings

def store_embeddings(client: weaviate.WeaviateClient, pdf_id: str, chunks_data: Union[List[Dict[str, any]], List[TextChunk]], embeddings: List[List[float]]):
    """
    Store embeddings in Weaviate.
    
    Args:
        client: Weaviate client
        pdf_id: PDF identifier
        chunks_data: Either a list of dictionaries or TextChunk objects
        embeddings: List of embedding vectors
    """
    class_name = config.WEAVIATE_CLASS_NAME
    collection = client.collections.get(class_name)
    
    data_objects = []
    for i, chunk in enumerate(chunks_data):
        # Handle both dictionary and TextChunk object formats
        if isinstance(chunk, TextChunk):
            properties = {
                "text": chunk.text,
                "pdf_id": pdf_id,
                "page_number": chunk.page_number
            }
        elif isinstance(chunk, dict):
            properties = {
                "text": chunk["text"],
                "pdf_id": pdf_id,
                "page_number": chunk.get("page_number", 0)
            }
        else:
            logger.error(f"Unexpected chunk type at index {i}: {type(chunk)}")
            continue
            
        if i < len(embeddings):
            vector = embeddings[i]
            
            if not isinstance(vector, list) or not all(isinstance(num, (int, float)) for num in vector):
                logger.error(f"Invalid vector format for chunk {i}: {type(vector)}")
                continue
            
            # Ensure all values are floats
            vector = [float(v) for v in vector]
            data_objects.append(wvc.data.DataObject(properties=properties, vector=vector))
        else:
            logger.error(f"No embedding found for chunk {i}")

    if data_objects:
        try:
            with collection.batch.dynamic() as batch:
                for data_obj in data_objects:
                    batch.add_object(
                        properties=data_obj.properties,
                        vector=data_obj.vector
                    )
            logger.info(f"Stored {len(data_objects)} chunks for pdf_id: {pdf_id} in Weaviate using batch.")
        except Exception as e:
            logger.error(f"Error during batch import to Weaviate: {e}", exc_info=True)
            raise
    else:
        logger.warning(f"No valid data objects to store for pdf_id: {pdf_id}")

def semantic_search(client: weaviate.WeaviateClient, query_text: str, pdf_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
    model = get_embedding_model()
    query_embedding = model.encode(query_text).tolist()

    collection = client.collections.get(config.WEAVIATE_CLASS_NAME)

    try:
        response = collection.query.near_vector(
            near_vector=query_embedding,
            filters=wvc.query.Filter.by_property("pdf_id").equal(pdf_id),
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True), 
            return_properties=["text", "page_number", "pdf_id"]
        )
        
        search_results = []
        for obj in response.objects:
            object_id = str(obj.uuid) if obj.uuid else None

            distance_val = None
            if obj.metadata and hasattr(obj.metadata, 'distance'):
                distance_val = obj.metadata.distance

            search_results.append({
                "text": obj.properties.get("text"),
                "page_number": obj.properties.get("page_number"),
                "pdf_id": obj.properties.get("pdf_id"),
                "_additional": {
                    "distance": distance_val,
                    "id": object_id 
                }
            })
        return search_results
    except WeaviateQueryException as e:
        logger.error(f"Weaviate query error for query '{query_text}', pdf_id '{pdf_id}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during semantic search for query '{query_text}', pdf_id '{pdf_id}': {e}", exc_info=True)
        return []