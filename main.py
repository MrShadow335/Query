# main.py - Updated with Gemini embedding-001, Pinecone, and PostgreSQL
import os
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.genai import types
import PyPDF2
import requests
import pinecone
from pinecone import Pinecone, ServerlessSpec
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import numpy as np
from dotenv import load_dotenv
import uvicorn
import hashlib

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HackRX 5.0 - LLM-Powered Query-Retrieval System",
    description="Intelligent Document Analysis API with Gemini Embedding-001, Pinecone & PostgreSQL",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security and middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]
)

security = HTTPBearer(auto_error=False)

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")

# Pydantic models
class HackRXRequest(BaseModel):
    documents: str = Field(..., description="PDF URL from blob storage")
    questions: List[str] = Field(..., description="Array of questions to answer")

class HackRXResponse(BaseModel):
    answers: List[str] = Field(..., description="Array of answers corresponding to questions")

# Enhanced authentication
async def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    valid_prefixes = ['hackrx_', 'api_', 'bearer_']
    if not any(token.startswith(prefix) for prefix in valid_prefixes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token

class PostgreSQLManager:
    """PostgreSQL database manager for document and query storage"""
    
    def __init__(self):
        self.connection_string = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_CONNECTION_STRING")
        if not self.connection_string:
            logger.warning("No PostgreSQL connection string provided")
            
    async def initialize_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    content_hash VARCHAR(64) NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            
            # Create query logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence_score FLOAT,
                    processing_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    async def store_document(self, url: str, content: str) -> int:
        """Store document in PostgreSQL and return document ID"""
        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Check if document already exists
            cursor.execute("SELECT id FROM documents WHERE url = %s", (url,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.close()
                conn.close()
                return existing[0]
            
            # Insert new document
            cursor.execute("""
                INSERT INTO documents (url, content, content_hash)
                VALUES (%s, %s, %s) RETURNING id
            """, (url, content, content_hash))
            
            document_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Document stored with ID: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise
    
    async def log_query(self, document_id: int, question: str, answer: str, 
                       confidence_score: float, processing_time: float):
        """Log query and response"""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO query_logs (document_id, question, answer, confidence_score, processing_time)
                VALUES (%s, %s, %s, %s, %s)
            """, (document_id, question, answer, confidence_score, processing_time))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log query: {e}")

class GeminiEmbeddingService:
    """Gemini embedding-001 service for vector embeddings"""
    
    def __init__(self):
        self.model_name = "gemini-embedding-001"
        self.embedding_dimension = 3072  # Default dimension for gemini-embedding-001[1]
        self.client = genai.Client() if hasattr(genai, 'Client') else None
        
    async def get_embeddings(self, texts: List[str], output_dimension: int = 1536) -> List[List[float]]:
        """Generate embeddings using Gemini embedding-001[1][2]"""
        try:
            embeddings = []
            
            # Process texts in batches to avoid API limits
            batch_size = 20  # Conservative batch size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                
                for text in batch:
                    try:
                        if self.client:
                            # Using new Google GenAI client[2]
                            result = self.client.models.embed_content(
                                model=self.model_name,
                                contents=text,
                                config=types.EmbedContentConfig(output_dimensionality=output_dimension)
                            )
                            embedding = result.embeddings[0].values
                        else:
                            # Fallback to older API
                            result = genai.embed_content(
                                model=self.model_name,
                                content=text,
                                output_dimensionality=output_dimension
                            )
                            embedding = result['embedding']
                        
                        batch_embeddings.append(embedding)
                        
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for text: {e}")
                        # Use zero vector as fallback
                        batch_embeddings.append([0.0] * output_dimension)
                
                embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            logger.info(f"Generated {len(embeddings)} embeddings using {self.model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

class PineconeVectorStore:
    """Pinecone vector store manager[6][9]"""
    
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")
        self.dimension = 1536  # Using reduced dimension for efficiency[2]
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
    async def initialize_index(self):
        """Initialize or connect to Pinecone index[6]"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx['name'] for idx in existing_indexes]
            
            if self.index_name not in index_names:
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-west-2'
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
                
                # Wait for index to be ready
                await asyncio.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise
    
    async def store_embeddings(self, document_id: str, chunks: List[str], embeddings: List[List[float]]):
        """Store document chunks and embeddings in Pinecone[6]"""
        try:
            if not self.index:
                await self.initialize_index()
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_chunk_{i}"
                metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk[:1000],  # Limit metadata size
                    "text_length": len(chunk)
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Stored {len(vectors)} vectors for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 5, document_id: str = None) -> List[Dict]:
        """Search for similar vectors[6]"""
        try:
            if not self.index:
                await self.initialize_index()
            
            # Build filter if document_id provided
            filter_dict = {"document_id": {"$eq": document_id}} if document_id else None
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Extract relevant information
            similar_chunks = []
            for match in results['matches']:
                if match['score'] > 0.7:  # Similarity threshold
                    similar_chunks.append({
                        "text": match['metadata']['text'],
                        "score": match['score'],
                        "chunk_index": match['metadata']['chunk_index']
                    })
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

class OptimizedDocumentProcessor:
    """Enhanced document processor with database integration"""
    
    def __init__(self, db_manager: PostgreSQLManager):
        self.db_manager = db_manager
        self.cache = {}
        self.session = requests.Session()
        
    async def process_pdf_url(self, pdf_url: str) -> tuple[str, int]:
        """Process PDF and return content with document ID"""
        # Check cache first
        if pdf_url in self.cache:
            return self.cache[pdf_url]
        
        try:
            logger.info(f"Processing PDF from: {pdf_url}")
            response = self.session.get(pdf_url, timeout=20, stream=True)
            response.raise_for_status()
            
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
            
            text = self._extract_text_from_bytes(content)
            
            # Store in database
            document_id = await self.db_manager.store_document(pdf_url, text)
            
            # Cache result
            self.cache[pdf_url] = (text, document_id)
            
            logger.info(f"Document processed. Length: {len(text)} chars, ID: {document_id}")
            return text, document_id
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
    def _extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            import io
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
            
            if not text.strip():
                raise Exception("No text content extracted from PDF")
            
            return text
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

class AdvancedQueryEngine:
    """Advanced query processing with Gemini, Pinecone & PostgreSQL"""
    
    def __init__(self, embedding_service: GeminiEmbeddingService, 
                 vector_store: PineconeVectorStore, db_manager: PostgreSQLManager):
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.db_manager = db_manager
        
    async def initialize_document(self, document_text: str, document_id: int):
        """Initialize document for querying"""
        try:
            # Create smart chunks
            chunks = self._create_smart_chunks(document_text)
            
            # Generate embeddings using Gemini embedding-001
            logger.info(f"Generating embeddings for {len(chunks)} chunks using Gemini embedding-001")
            embeddings = await self.embedding_service.get_embeddings(chunks)
            
            # Store in Pinecone
            await self.vector_store.store_embeddings(str(document_id), chunks, embeddings)
            
            logger.info("Document initialized successfully")
            
        except Exception as e:
            logger.error(f"Document initialization failed: {e}")
            raise
    
    async def answer_questions(self, questions: List[str], document_id: int) -> List[str]:
        """Process multiple questions with logging"""
        answers = []
        
        for i, question in enumerate(questions):
            start_time = time.time()
            try:
                logger.info(f"Processing question {i+1}/{len(questions)}")
                
                # Generate query embedding
                query_embeddings = await self.embedding_service.get_embeddings([question])
                query_embedding = query_embeddings[0]
                
                # Search for relevant chunks
                similar_chunks = await self.vector_store.search_similar(
                    query_embedding, top_k=5, document_id=str(document_id)
                )
                
                # Generate answer
                answer, confidence = await self._generate_answer_with_confidence(question, similar_chunks)
                answers.append(answer)
                
                # Log query
                processing_time = time.time() - start_time
                await self.db_manager.log_query(
                    document_id, question, answer, confidence, processing_time
                )
                
            except Exception as e:
                logger.error(f"Failed to answer question {i+1}: {e}")
                answers.append("Unable to process this question due to technical error.")
        
        return answers
    
    def _create_smart_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Create overlapping chunks with sentence boundaries"""
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text + '.')
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text + '.')
        
        return chunks
    
    async def _generate_answer_with_confidence(self, question: str, similar_chunks: List[Dict]) -> tuple[str, float]:
        """Generate answer with confidence score"""
        if not similar_chunks:
            return "Information not available in the provided document.", 0.0
        
        # Extract context from similar chunks
        context = "\n\n".join([chunk['text'] for chunk in similar_chunks[:3]])
        avg_similarity = sum(chunk['score'] for chunk in similar_chunks[:3]) / len(similar_chunks[:3])
        
        prompt = f"""Based on the following document context, provide a direct and accurate answer to the question.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, factual answer based ONLY on the information in the context
- If the information is not available in the context, respond with "Information not available in the document"
- Be specific about numbers, dates, conditions, and requirements when they appear in the context
- Keep the answer concise but complete

ANSWER:"""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=300,
                    temperature=0.1,
                    top_p=0.9
                )
            )
            
            answer = response.text.strip()
            confidence = min(avg_similarity, 1.0)  # Normalize confidence
            
            return answer if answer else "Unable to generate answer from the provided context.", confidence
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer due to technical error.", 0.0

# Initialize components
db_manager = PostgreSQLManager()
embedding_service = GeminiEmbeddingService()
vector_store = PineconeVectorStore()
doc_processor = OptimizedDocumentProcessor(db_manager)
query_engine = AdvancedQueryEngine(embedding_service, vector_store, db_manager)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await db_manager.initialize_database()
        await vector_store.initialize_index()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Startup initialization failed: {e}")

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_main_endpoint(
    request: HackRXRequest,
    token: str = Depends(verify_bearer_token)
):
    """Main HackRX 5.0 API endpoint with Gemini embedding-001, Pinecone & PostgreSQL"""
    request_start = time.time()
    
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Step 1: Process PDF document
        document_text, document_id = await doc_processor.process_pdf_url(request.documents)
        
        # Step 2: Initialize document for querying
        await query_engine.initialize_document(document_text, document_id)
        
        # Step 3: Process all questions
        answers = await query_engine.answer_questions(request.questions, document_id)
        
        total_time = time.time() - request_start
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        total_time = time.time() - request_start
        logger.error(f"Error after {total_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "gemini_embedding": "gemini-embedding-001",
            "vector_store": "pinecone",
            "database": "postgresql",
            "llm": "gemini-1.5-flash"
        }
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "HackRX 5.0 - Advanced LLM-Powered Query-Retrieval System",
        "version": "2.0.0",
        "tech_stack": {
            "embeddings": "Gemini embedding-001",
            "vector_store": "Pinecone",
            "database": "PostgreSQL",
            "llm": "Gemini 1.5 Flash",
            "backend": "FastAPI"
        },
        "features": [
            "Gemini embedding-001 for superior embeddings",
            "Pinecone vector store for scalable similarity search",
            "PostgreSQL for persistent data storage",
            "Advanced query logging and analytics"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
