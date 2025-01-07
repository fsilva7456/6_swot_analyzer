import os
import logging
from contextlib import asynccontextmanager
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
   # Startup
   logger.info("Starting up application...")
   logger.info("Checking environment variables...")
   if not all([os.getenv('OPENAI_API_KEY'), os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')]):
       logger.error("Missing required environment variables!")
   else:
       logger.info("All required environment variables are set")
   yield
   # Shutdown
   logger.info("Shutting down application...")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Supabase client
supabase: Client = create_client(
   os.getenv('SUPABASE_URL'),
   os.getenv('SUPABASE_KEY')
)

class SWOTAnalysis(BaseModel):
   strengths: str
   weaknesses: str
   opportunities: str
   threats: str

class CompetitorResponse(BaseModel):
   id: int
   competitor_name: str
   competitor_strength: Optional[str] 
   competitor_weakness: Optional[str]
   competitor_opportunity: Optional[str]
   competitor_threats: Optional[str]

def get_swot_analysis(competitor_name: str, existing_data: Dict) -> SWOTAnalysis:
   """
   Use OpenAI to perform SWOT analysis using existing competitor data
   """
   logger.info(f"Performing SWOT analysis for {competitor_name}")
   prompt = f"""Perform a detailed SWOT analysis of {competitor_name}'s loyalty program based on this existing data:

   Program Summary: {existing_data.get('program_summary', 'N/A')}
   Market Positioning: {existing_data.get('competitor_positioning', 'N/A')}
   Rewards and Benefits: {existing_data.get('competitor_rewards_benefits', 'N/A')}
   User Feedback: {existing_data.get('competitor_user_feedback', 'N/A')}

   Provide a comprehensive SWOT analysis with each component in 3 bullet points."""

   completion = client.beta.chat.completions.parse(
       model="gpt-4o-2024-11-20",
       messages=[
           {"role": "system", "content": "You are a helpful assistant that performs SWOT analysis of loyalty programs."},
           {"role": "user", "content": prompt}
       ],
       response_format=SWOTAnalysis
   )
   
   return completion.choices[0].message.parsed

def update_competitor_swot(competitor_id: int, competitor_name: str, existing_data: Dict) -> CompetitorResponse:
   """
   Update a competitor's row with SWOT analysis
   """
   logger.info(f"Updating SWOT analysis for competitor ID {competitor_id}: {competitor_name}")
   try:
       swot = get_swot_analysis(competitor_name, existing_data)
       
       response = supabase.table('competitors').update({
           'competitor_strength': swot.strengths,
           'competitor_weakness': swot.weaknesses,
           'competitor_opportunity': swot.opportunities,
           'competitor_threats': swot.threats
       }).eq('id', competitor_id).execute()
       
       updated_competitor = response.data[0]
       logger.info(f"Successfully updated SWOT analysis for {competitor_name}")
       
       return CompetitorResponse(**updated_competitor)
   except Exception as e:
       logger.error(f"Error processing {competitor_name}: {str(e)}")
       raise

@app.get("/")
async def root():
   logger.info("Health check endpoint called")
   return {"status": "API is running"}

@app.post("/update-single/{competitor_id}")
async def update_single_competitor(competitor_id: int):
   """
   Update SWOT analysis for a single competitor
   """
   try:
       logger.info(f"Received request to update competitor ID: {competitor_id}")
       
       # Get competitor details with all existing data
       response = supabase.table('competitors').select(
           'id, competitor_name, program_summary, competitor_positioning, competitor_rewards_benefits, competitor_user_feedback'
       ).eq('id', competitor_id).execute()
       
       if not response.data:
           logger.error(f"No competitor found with ID {competitor_id}")
           raise HTTPException(status_code=404, detail="Competitor not found")
           
       competitor = response.data[0]
       updated_competitor = update_competitor_swot(
           competitor_id,
           competitor['competitor_name'],
           competitor
       )
       
       return updated_competitor
       
   except Exception as e:
       logger.error(f"Error processing request: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-all")
async def update_all_competitors():
   """
   Update SWOT analysis for all competitors without analyses
   """
   try:
       logger.info("Starting batch update of all competitors")
       response = supabase.table('competitors').select(
           'id, competitor_name, program_summary, competitor_positioning, competitor_rewards_benefits, competitor_user_feedback'
       ).is_('competitor_strength', 'null').execute()
       
       if not response.data:
           logger.info("No competitors found needing SWOT analysis")
           return {"status": "No competitors found needing updates"}
       
       logger.info(f"Found {len(response.data)} competitors to process")
       updated_competitors = []
       
       for competitor in response.data:
           try:
               updated = update_competitor_swot(
                   competitor['id'],
                   competitor['competitor_name'],
                   competitor
               )
               updated_competitors.append(updated)
               logger.info(f"Successfully processed {competitor['competitor_name']}")
           except Exception as e:
               logger.error(f"Error processing {competitor['competitor_name']}: {str(e)}")
       
       return {
           "status": "success",
           "total_processed": len(updated_competitors),
           "updated_competitors": updated_competitors
       }
       
   except Exception as e:
       logger.error(f"Error in batch update: {str(e)}")
       raise HTTPException(status_code=500, detail=str(e))
