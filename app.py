"""
Customer Call Transcript Analyzer
A FastAPI application that analyzes customer call transcripts using Groq API
for summarization and sentiment analysis.
"""

import csv
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from groq import AsyncGroq
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "g******************************************")  #use you real key 
CSV_OUTPUT_FILE = "call_analysis.csv"
GROQ_MODEL = "llama-3.1-8b-instant"


Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Initialize Groq client
groq_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global groq_client
    try:
        groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise
    finally:
        logger.info("Application shutdown")

# Initialize FastAPI app
app = FastAPI(
    title="Customer Call Transcript Analyzer",
    description="Analyze customer call transcripts for summary and sentiment using Groq AI",
    version="1.0.0",
    lifespan=lifespan
)

# templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models
class TranscriptRequest(BaseModel):
    """Request model for transcript analysis"""
    transcript: str = Field(..., min_length=10, max_length=10000, description="Customer call transcript")
    
    @field_validator('transcript')
    @classmethod
    def validate_transcript(cls, v):
        if not v.strip():
            raise ValueError('Transcript cannot be empty')
        return v.strip()

class AnalysisResponse(BaseModel):
    """Response model for transcript analysis"""
    transcript: str
    summary: str
    sentiment: str
    timestamp: str
    success: bool = True

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    success: bool = False

# Business Logic
class TranscriptAnalyzer:
    """Handles transcript analysis using Groq API"""
    
    def __init__(self, client: AsyncGroq):
        self.client = client
        self.model = GROQ_MODEL
    
    async def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze transcript for summary and sentiment
        """
        try:
            #summary
            summary = await self._get_summary(transcript)
            
            #sentiment
            sentiment = await self._get_sentiment(transcript)
            
            return {
                "summary": summary,
                "sentiment": sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transcript: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def _get_summary(self, transcript: str) -> str:
        """Generate summary using Groq API"""
        prompt = f"""
        Summarize the following customer call transcript in 2-3 clear, concise sentences.
        Focus on the main issue, customer concern, and any resolution discussed.
        
        Transcript: {transcript}
        
        Summary:
        """
        
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional customer service analyst. Provide clear, concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return "Unable to generate summary due to API error."
    
    async def _get_sentiment(self, transcript: str) -> str:
        """Extract sentiment using Groq API"""
        prompt = f"""
        Analyze the customer sentiment in the following transcript.
        Classify it as exactly one of: Positive, Neutral, or Negative.
        Consider the tone, language, and overall customer satisfaction.
        
        Transcript: {transcript}
        
        Sentiment (one word only - Positive/Neutral/Negative):
        """
        
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Always respond with exactly one word: Positive, Neutral, or Negative."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=10,
                temperature=0.1
            )
            
            sentiment = response.choices[0].message.content.strip()
            
            # Validate and normalize sentiment
            if sentiment.lower() in ['positive', 'neutral', 'negative']:
                return sentiment.capitalize()
            else:
              
                return "Neutral"
                
        except Exception as e:
            logger.error(f"Error getting sentiment: {e}")
            return "Neutral"

class CSVManager:
    """Handles CSV file operations"""
    
    @staticmethod
    def save_to_csv(data: Dict[str, Any], filename: str = CSV_OUTPUT_FILE):
        """Save analysis data to CSV file"""
        try:
            file_exists = Path(filename).exists()
            
            with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Timestamp', 'Transcript', 'Summary', 'Sentiment']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
               
                if not file_exists:
                    writer.writeheader()
                
                # Write data
                writer.writerow({
                    'Timestamp': data['timestamp'],
                    'Transcript': data['transcript'],
                    'Summary': data['summary'],
                    'Sentiment': data['sentiment']
                })
                
            logger.info(f"Data saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return False
    
    @staticmethod
    def read_csv(filename: str = CSV_OUTPUT_FILE) -> Optional[pd.DataFrame]:
        """Read CSV file and return as DataFrame"""
        try:
            if Path(filename).exists():
                return pd.read_csv(filename)
            return None
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return None

# Initialize analyzer
analyzer = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_transcript_api(request: TranscriptRequest):
    """
    API endpoint to analyze customer call transcript
    """
    global analyzer
    
    if not analyzer:
        analyzer = TranscriptAnalyzer(groq_client)
    
    try:
        # Analyze 
        result = await analyzer.analyze_transcript(request.transcript)
        
        # Prepare response data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_data = {
            "transcript": request.transcript,
            "summary": result["summary"],
            "sentiment": result["sentiment"],
            "timestamp": timestamp
        }
        
        # Save to CSV
        csv_saved = CSVManager.save_to_csv(response_data)
        if not csv_saved:
            logger.warning("Failed to save data to CSV")
        
        # Log the analysis
        logger.info(f"Analysis completed - Sentiment: {result['sentiment']}")
        
        return AnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_transcript_api: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze-form")
async def analyze_transcript_form(request: Request, transcript: str = Form(...)):
    """
    Form-based endpoint for web interface
    """
    global analyzer
    
    if not analyzer:
        analyzer = TranscriptAnalyzer(groq_client)
    
    try:
        # Validate input
        if len(transcript.strip()) < 10:
            raise ValueError("Transcript must be at least 10 characters long")
        
      
        result = await analyzer.analyze_transcript(transcript)
        
        # Prepare response 
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_data = {
            "transcript": transcript,
            "summary": result["summary"],
            "sentiment": result["sentiment"],
            "timestamp": timestamp
        }
        
        # Save to CSV
        CSVManager.save_to_csv(response_data)
        
        # Return results 
        return templates.TemplateResponse("results.html", {
            "request": request,
            "data": response_data
        })
        
    except Exception as e:
        logger.error(f"Error in form analysis: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/results")
async def view_results(request: Request):
    """View all analysis results"""
    try:
        df = CSVManager.read_csv()
        if df is not None and not df.empty:
            
            results = df.to_dict('records')
            return templates.TemplateResponse("all_results.html", {
                "request": request,
                "results": results
            })
        else:
            return templates.TemplateResponse("all_results.html", {
                "request": request,
                "results": [],
                "message": "No analysis results found."
            })
    except Exception as e:
        logger.error(f"Error viewing results: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Failed to load results: {str(e)}"
        })

@app.get("/health")      #health
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/results")
async def get_results_api():
    """API endpoint to get all results as JSON"""
    try:
        df = CSVManager.read_csv()
        if df is not None and not df.empty:
            return {"results": df.to_dict('records'), "count": len(df)}
        else:
            return {"results": [], "count": 0}
    except Exception as e:
        logger.error(f"Error getting results via API: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve results")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )