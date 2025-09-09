from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
from mcp_server import main as mcp_main
import threading
import time
from typing import List, Dict
from components.search.search import perform_search, get_all_events, initialize_embeddings, refresh_embeddings

from components.Event_ai.crew import EventContentCrew

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Semantic Event Search API")

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

# -----------------------------
# Background task for periodic refresh
# -----------------------------
# def periodic_refresh():
#     """Refresh embeddings every 30 minutes"""
#     while True:
#         time.sleep(1800)  # 30 minutes
#         try:
#             logger.info("Starting periodic embedding refresh...")
#             refresh_embeddings()
#             logger.info("Embeddings refreshed successfully")
#         except Exception as e:
#             logger.error(f"Error during periodic refresh: {e}")

# Start background thread on app startup
@app.on_event("startup")
async def startup_event():
    # Initialize embeddings
    try:
        initialize_embeddings()
        logger.info("Embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
    
    # Start background refresh thread
    # refresh_thread = threading.Thread(target=periodic_refresh, daemon=True)
    # refresh_thread.start()
    # logger.info("Background refresh thread started")

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
    # print(result)
    return {"proposals": result["proposals"]}


@app.post("/refresh-embeddings")
async def manual_refresh():
    """Manual endpoint to trigger embedding refresh"""
    try:
        refresh_embeddings()
        return {"status": "success", "message": "Embeddings refreshed successfully"}
    except Exception as e:
        logger.error(f"Manual refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    asyncio.run(mcp_main())