from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import threading
import time
import asyncio
from typing import Dict

# Import core components
from components.search.search import perform_search, get_all_events, initialize_embeddings, refresh_embeddings
from components.Event_ai.crew import EventContentCrew

# Import MCP server
from mcp_server import main as mcp_main

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="EventHub API",
    description="Semantic Event Search, Event Creation, and AI-assisted tools API",
    version="1.1.0",
    docs_url="/docs"
)

# -----------------------------
# Middleware
# -----------------------------
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
    title: str

# -----------------------------
# Background task for periodic refresh
# -----------------------------
def periodic_refresh():
    """Refresh embeddings every 30 minutes"""
    while True:
        time.sleep(1800)
        try:
            logger.info("Starting periodic embedding refresh...")
            refresh_embeddings()
            logger.info("Embeddings refreshed successfully")
        except Exception as e:
            logger.error(f"Error during periodic refresh: {e}")

# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    try:
        initialize_embeddings()
        logger.info("Embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")

    # Start background refresh thread
    refresh_thread = threading.Thread(target=periodic_refresh, daemon=True)
    refresh_thread.start()
    logger.info("Background refresh thread started")

    # Launch MCP server in background
    asyncio.create_task(mcp_main())
    logger.info("MCP server started in background")

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "EventHub API is running"}

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    try:
        results = perform_search(request.query, request.user_id)
        if not results:
            return {"results": [], "message": "No matching events found"}
        return {"results": results}
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/generate-event")
def generate_event(event: EventInput):
    try:
        crew = EventContentCrew()
        result = crew.eventcrew().kickoff(
            inputs={"event_description": event.description, "title": event.title}
        )
        return {"proposals": result.get("proposals", [])}
    except Exception as e:
        logger.error(f"Generate-event error: {e}")
        raise HTTPException(status_code=500, detail="Event generation failed")

@app.post("/refresh-embeddings")
async def manual_refresh():
    try:
        refresh_embeddings()
        return {"status": "success", "message": "Embeddings refreshed successfully"}
    except Exception as e:
        logger.error(f"Manual refresh error: {e}")
        raise HTTPException(status_code=500, detail="Embedding refresh failed")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting EventHub server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)