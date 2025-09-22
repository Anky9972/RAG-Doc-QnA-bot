from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from contextlib import asynccontextmanager # For lifespan

from app.core import config
from app.schemas import QueryRequest, QueryResponse, UploadResponse
from app.services import pdf_processor, vector_store, llm_handler
import weaviate # For type hinting Depends

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Application startup via lifespan...")
    logger.info(f"Attempting to initialize Weaviate client with URL: {config.WEAVIATE_URL}")
    try:
        client = vector_store.get_weaviate_client()
        if client:
             logger.info("Weaviate client initialized and schema checked/created successfully.")
        else:
            logger.error("Failed to initialize Weaviate client on startup.")
        
        vector_store.get_embedding_model()
        logger.info("Embedding model loaded.")
        
        llm_handler.get_gemini_model()
        # The get_gemini_model will log its success or failure
    except Exception as e:
        logger.error(f"Error during startup initialization via lifespan: {e}", exc_info=True)
        # Depending on the error, you might want to prevent the app from fully starting
        # For now, it logs and continues.
    
    yield # Application runs here

    logger.info("Application shutdown via lifespan...")
    global_weaviate_client = getattr(vector_store, 'weaviate_client', None)
    if global_weaviate_client and hasattr(global_weaviate_client, 'is_connected') and global_weaviate_client.is_connected():
        logger.info("Closing Weaviate client.")
        try:
            global_weaviate_client.close()
            logger.info("Weaviate client closed.")
        except Exception as e:
            logger.error(f"Error closing Weaviate client during shutdown: {e}", exc_info=True)
    else:
        logger.info("Weaviate client was not connected or not initialized; no close action taken.")

app = FastAPI(title="PDF Q&A System with Gemini & Weaviate", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_client():
    try:
        client = vector_store.get_weaviate_client()
        if client is None or not client.is_connected(): # Check if client is connected
            logger.error("Weaviate client is not available or not connected.")
            raise HTTPException(status_code=503, detail="Weaviate service unavailable or not connected.")
        return client
    except ConnectionError as e: # This might be raised by get_weaviate_client on init failure
        logger.error(f"Weaviate connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Weaviate service unavailable: {e}")
    except Exception as e: # Catch other unexpected errors during client retrieval
        logger.error(f"Unexpected error getting Weaviate client: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error initializing or getting Weaviate: {e}")

@app.post("/upload_pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), db_client: weaviate.WeaviateClient = Depends(get_db_client)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    try:
        pdf_id = str(uuid.uuid4()) 
        logger.info(f"Processing PDF: {file.filename} with pdf_id: {pdf_id}")
        
        contents = await file.read()
        
        logger.info(f"Read {len(contents)} bytes from {file.filename}. Splitting into chunks...")
        chunks = pdf_processor.load_and_split_pdf(contents)
        
        if not chunks:
            logger.warning(f"No text chunks extracted from {file.filename}.")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF or PDF is empty.")

        logger.info(f"Extracted {len(chunks)} chunks. Embedding chunks...")
        embeddings = vector_store.embed_chunks(chunks)
        
        if not embeddings or len(embeddings) != len(chunks):
            logger.error("Embedding process failed or produced inconsistent number of embeddings.")
            raise HTTPException(status_code=500, detail="Failed to embed PDF chunks.")

        logger.info(f"Storing {len(embeddings)} embeddings in Weaviate for pdf_id: {pdf_id}...")
        vector_store.store_embeddings(db_client, pdf_id, chunks, embeddings)
        
        logger.info(f"Successfully processed and stored {file.filename} with pdf_id: {pdf_id}")
        return UploadResponse(
            message="PDF processed and embeddings stored successfully.",
            pdf_id=pdf_id,
            filename=file.filename
        )
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Error processing PDF {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def query_pdf(request: QueryRequest, db_client: weaviate.WeaviateClient = Depends(get_db_client)):
    try:
        logger.info(f"Received query: '{request.query}' for pdf_id: '{request.pdf_id}'")
        
        logger.info("Performing semantic search for relevant chunks...")
        relevant_chunks = vector_store.semantic_search(db_client, request.query, request.pdf_id, top_k=5)
        
        # Query LLM even if no chunks are found, to give a consistent "not found" message from LLM
        answer = llm_handler.query_llm(request.query, relevant_chunks) # relevant_chunks will be [] if none found
        
        sources_for_response = []
        if relevant_chunks:
            logger.info(f"Found {len(relevant_chunks)} relevant chunks.")
            sources_for_response = [
                {"text": chunk.get("text", ""), "page_number": chunk.get("page_number")}
                for chunk in relevant_chunks
            ]
        else:
            logger.info("No relevant chunks found for the query by semantic search.")
            # The LLM will handle saying "I could not find..." based on empty context

        logger.info(f"LLM processing for query '{request.query}' complete.")
        return QueryResponse(answer=answer, sources=sources_for_response)
    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # This is for running directly, e.g. `python app/main.py`
    # For production, use `uvicorn app.main:app --host 0.0.0.0 --port 8000` (without --reload)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) # Correct way to call uvicorn.run for reload