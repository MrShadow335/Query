# main.py - Updated to match client specifications
import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import PyPDF2
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Query-Retrieval System", 
    version="1.0.0",
    description="HackRX 5.0 Submission - Intelligent Document Query System"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Request/Response Models matching client specifications
class HackRXRequest(BaseModel):
    documents: str  # PDF URL as specified in the format
    questions: List[str]  # Array of questions

class HackRXResponse(BaseModel):
    answers: List[str]  # Array of answers corresponding to questions

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token - implement your token validation logic here"""
    token = credentials.credentials
    # For demo purposes, accepting any token that starts with 'hackrx_'
    if not token.startswith('hackrx_'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

class DocumentProcessor:
    """Enhanced document processor with caching and optimization"""
    
    def __init__(self):
        self.document_cache = {}
    
    async def extract_text_from_pdf_url(self, pdf_url: str) -> str:
        """Async document processing with caching"""
        if pdf_url in self.document_cache:
            logger.info("Using cached document")
            return self.document_cache[pdf_url]
        
        try:
            # Download PDF with timeout
            response = requests.get(pdf_url, timeout=25)  # Leave 5s for processing
            response.raise_for_status()
            
            # Process PDF
            temp_path = f"temp_{datetime.now().timestamp()}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            text = ""
            with open(temp_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Cleanup and cache
            os.remove(temp_path)
            self.document_cache[pdf_url] = text
            
            logger.info(f"Document processed successfully. Length: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

class IntelligentQueryProcessor:
    """Main processing engine optimized for speed and accuracy"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedding_model = embedding_model
        self.document_embeddings = None
        self.text_chunks = []
    
    async def setup_document(self, document_text: str):
        """Setup document embeddings for fast querying"""
        self.text_chunks = self._split_text_optimized(document_text)
        
        # Create embeddings
        embeddings = self.embedding_model.encode(self.text_chunks)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    async def process_multiple_questions(self, questions: List[str]) -> List[str]:
        """Process multiple questions efficiently"""
        answers = []
        
        for question in questions:
            try:
                # Fast retrieval of relevant context
                relevant_chunks = self._get_relevant_context(question)
                context = "\n\n".join(relevant_chunks[:3])  # Top 3 most relevant
                
                # Generate answer with optimized prompt
                answer = await self._generate_answer_optimized(question, context)
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append("Unable to process this question due to technical limitations.")
        
        return answers
    
    def _get_relevant_context(self, question: str, top_k: int = 5) -> List[str]:
        """Fast context retrieval using FAISS"""
        if not hasattr(self, 'index') or self.index is None:
            return []
        
        # Encode question
        query_embedding = self.embedding_model.encode([question])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.text_chunks) and scores[0][i] > 0.25:
                relevant_chunks.append(self.text_chunks[idx])
        
        return relevant_chunks
    
    async def _generate_answer_optimized(self, question: str, context: str) -> str:
        """Generate answer with optimized prompt for speed and accuracy"""
        prompt = f"""Based on the document context provided, answer the question directly and concisely.

Context:
{context}

Question: {question}

Instructions:
- Provide a direct, factual answer
- If the information is not in the context, state "Information not available in the document"
- Be specific about coverage, conditions, or requirements when applicable
- Keep the answer concise but complete

Answer:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,  # Limit for faster response
                    temperature=0.1,  # Low temperature for consistency
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Unable to generate answer due to technical error."
    
    def _split_text_optimized(self, text: str, chunk_size: int = 400) -> List[str]:
        """Optimized text splitting for better context preservation"""
        # Split by sentences first, then by chunks
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks

# Initialize components
doc_processor = DocumentProcessor()
query_processor = IntelligentQueryProcessor()

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_main_endpoint(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """
    Main HackRX endpoint - Processes documents and questions as per client specifications
    
    Expected format:
    {
        "documents": "https://example.com/policy.pdf",
        "questions": ["Question 1", "Question 2", ...]
    }
    
    Returns:
    {
        "answers": ["Answer 1", "Answer 2", ...]
    }
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Step 1: Extract document content (Component 1)
        document_text = await doc_processor.extract_text_from_pdf_url(request.documents)
        
        # Step 2: Setup document for querying (Components 2-3)
        await query_processor.setup_document(document_text)
        
        # Step 3: Process all questions (Components 4-6)
        answers = await query_processor.process_multiple_questions(request.questions)
        
        # Ensure response time < 30s
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request processed in {processing_time:.2f} seconds")
        
        if processing_time > 28:  # Warning if close to limit
            logger.warning("Processing time approaching 30s limit")
        
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error after {processing_time:.2f}s: {str(e)}")
        
        # Return partial results if possible
        if 'answers' in locals() and answers:
            return HackRXResponse(answers=answers)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "ready": True
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "hackrx": "5.0",
        "team": "Your Team Name",
        "tech_stack": "FastAPI + Gemini + FAISS + Pinecone",
        "endpoints": {
            "main": "POST /hackrx/run",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    # Production configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="path/to/key.pem",  # Add your SSL certificate
        ssl_certfile="path/to/cert.pem",  # Add your SSL certificate
        access_log=True,
        loop="asyncio"
    )
