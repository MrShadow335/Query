# main.py
import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import PyPDF2
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LLM-Powered Query-Retrieval System", version="1.0.0")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    query: str
    pdf_url: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    relevant_clauses: List[str]
    confidence_score: float
    reasoning: str
    timestamp: str

class DocumentProcessor:
    """Component 1: Input Documents - PDF Blob URL processor"""
    
    @staticmethod
    def extract_text_from_pdf_url(pdf_url: str) -> str:
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Save temporarily
            temp_path = f"temp_{datetime.now().timestamp()}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Extract text
            text = ""
            with open(temp_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Cleanup
            os.remove(temp_path)
            return text
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

class LLMParser:
    """Component 2: LLM Parser - Extract structured query using Gemini"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def parse_query(self, query: str, document_text: str) -> Dict[str, Any]:
        prompt = f"""
        You are an intelligent document analysis system. Analyze the following query and document to extract key information.
        
        Query: "{query}"
        
        Document excerpt: "{document_text[:2000]}..."
        
        Please provide a structured analysis in JSON format with:
        1. "intent": The main intent of the query
        2. "key_terms": Important terms to search for
        3. "question_type": Type of question (coverage, conditions, eligibility, etc.)
        4. "search_keywords": Keywords for semantic search
        
        Respond only with valid JSON.
        """
        
        try:
            response = self.model.generate_content(prompt)
            parsed_data = json.loads(response.text.strip())
            return parsed_data
        except Exception as e:
            logger.error(f"Error parsing query with Gemini: {str(e)}")
            return {
                "intent": "general_inquiry",
                "key_terms": query.split(),
                "question_type": "general",
                "search_keywords": query.split()
            }

class EmbeddingSearch:
    """Component 3: Embedding Search - FAISS/Pinecone retrieval"""
    
    def __init__(self):
        self.index = None
        self.text_chunks = []
        self.embedding_model = embedding_model
    
    def create_embeddings(self, text: str, chunk_size: int = 500) -> None:
        # Split text into chunks
        self.text_chunks = self._split_text(text, chunk_size)
        
        # Create embeddings
        embeddings = self.embedding_model.encode(self.text_chunks)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return relevant chunks
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.text_chunks) and scores[0][i] > 0.3:  # Similarity threshold
                relevant_chunks.append(self.text_chunks[idx])
        
        return relevant_chunks
    
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class ClauseMatching:
    """Component 4: Clause Matching - Semantic similarity"""
    
    @staticmethod
    def match_clauses(query_data: Dict[str, Any], relevant_chunks: List[str]) -> List[str]:
        key_terms = query_data.get("key_terms", [])
        matched_clauses = []
        
        for chunk in relevant_chunks:
            # Simple keyword matching with semantic understanding
            chunk_lower = chunk.lower()
            matches = 0
            
            for term in key_terms:
                if isinstance(term, str) and term.lower() in chunk_lower:
                    matches += 1
            
            # Include chunk if it has significant matches or mentions policy/coverage
            if (matches > 0 or 
                any(word in chunk_lower for word in ['policy', 'coverage', 'benefit', 'condition', 'exclusion'])):
                matched_clauses.append(chunk)
        
        return matched_clauses[:3]  # Return top 3 most relevant clauses

class LogicEvaluation:
    """Component 5: Logic Evaluation - Decision processing"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def evaluate_query(self, query: str, relevant_clauses: List[str], query_data: Dict[str, Any]) -> Dict[str, Any]:
        context = "\n\n".join(relevant_clauses)
        
        prompt = f"""
        You are an expert document analyst specializing in insurance, legal, HR, and compliance domains.
        
        Query: "{query}"
        
        Relevant Document Context:
        {context}
        
        Query Analysis: {json.dumps(query_data)}
        
        Based on the provided context, answer the query with:
        1. A clear, direct answer
        2. Confidence score (0.0 to 1.0)
        3. Step-by-step reasoning
        4. Any limitations or assumptions
        
        Be specific about coverage, conditions, exclusions, or requirements mentioned in the context.
        If the context doesn't contain enough information, clearly state this.
        
        Provide your response in the following JSON format:
        {{
            "answer": "Direct answer to the query",
            "confidence_score": 0.85,
            "reasoning": "Step-by-step explanation of how you arrived at this answer",
            "limitations": "Any limitations or missing information"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip())
            return result
        except Exception as e:
            logger.error(f"Error in logic evaluation: {str(e)}")
            return {
                "answer": "Unable to process query due to technical error.",
                "confidence_score": 0.0,
                "reasoning": f"Error occurred during processing: {str(e)}",
                "limitations": "Technical error prevented proper analysis"
            }

class JSONOutput:
    """Component 6: JSON Output - Structured response"""
    
    @staticmethod
    def format_response(query: str, evaluation_result: Dict[str, Any], 
                       relevant_clauses: List[str]) -> QueryResponse:
        return QueryResponse(
            query=query,
            answer=evaluation_result.get("answer", "No answer available"),
            relevant_clauses=relevant_clauses,
            confidence_score=evaluation_result.get("confidence_score", 0.0),
            reasoning=evaluation_result.get("reasoning", "No reasoning provided"),
            timestamp=datetime.now().isoformat()
        )

# Initialize system components
doc_processor = DocumentProcessor()
llm_parser = LLMParser()
embedding_search = EmbeddingSearch()
clause_matcher = ClauseMatching()
logic_evaluator = LogicEvaluation()
json_formatter = JSONOutput()

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main API endpoint that processes the query through all 6 components
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Component 1: Extract text from PDF
        document_text = doc_processor.extract_text_from_pdf_url(request.pdf_url)
        logger.info("Document text extracted successfully")
        
        # Component 2: Parse query with LLM
        query_data = llm_parser.parse_query(request.query, document_text)
        logger.info("Query parsed successfully")
        
        # Component 3: Create embeddings and search
        embedding_search.create_embeddings(document_text)
        relevant_chunks = embedding_search.search(request.query)
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Component 4: Match clauses
        matched_clauses = clause_matcher.match_clauses(query_data, relevant_chunks)
        logger.info(f"Matched {len(matched_clauses)} clauses")
        
        # Component 5: Evaluate with logic
        evaluation_result = logic_evaluator.evaluate_query(
            request.query, matched_clauses, query_data
        )
        logger.info("Logic evaluation completed")
        
        # Component 6: Format JSON output
        response = json_formatter.format_response(
            request.query, evaluation_result, matched_clauses
        )
        logger.info("Response formatted successfully")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
