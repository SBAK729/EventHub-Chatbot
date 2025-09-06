
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from typing import List, Dict
from components.search.search import perform_search, get_all_events

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

@app.get("/events")
async def events_endpoint():
    return {"events": get_all_events()}

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
