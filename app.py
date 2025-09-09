# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import uvicorn
from typing import Dict

from mcp_server import main as mcp_main   # your MCP server loop
from components.search.search import perform_search, get_all_events, initialize_embeddings, refresh_embeddings
from components.Event_ai.crew import EventContentCrew
from pydantic import BaseModel

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Semantic Event Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    user_id: str

class EventInput(BaseModel):
    description: str

# -----------------------------
# Startup events
# -----------------------------
@app.on_event("startup")
async def startup_event():
    # Initialize embeddings
    try:
        initialize_embeddings()
        logger.info("Embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")

    # Start MCP server in background
    loop = asyncio.get_event_loop()
    loop.create_task(mcp_main())
    logger.info("MCP server started in background")

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Semantic Event Search API is running"}

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    try:
        results = perform_search(request.query, request.user_id)
        return {"results": results}
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-event")
def generate_event(event: EventInput):
    crew = EventContentCrew()
    result = crew.eventcrew().kickoff(inputs={"event_description": event.description})
    return {"proposals": result["proposals"]}

@app.post("/refresh-embeddings")
async def manual_refresh():
    try:
        refresh_embeddings()
        return {"status": "success", "message": "Embeddings refreshed successfully"}
    except Exception as e:
        logger.error(f"Manual refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Main entrypoint
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting API + MCP server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
