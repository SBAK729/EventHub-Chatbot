# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Import search functions
from components.search.search import perform_search, get_search_history, get_all_events

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    n_results: int = 10

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Semantic Event Search API is running"}

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    try:
        return perform_search(request.query, request.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search-history")
async def search_history_endpoint(limit: int = 10):
    return {"history": get_search_history(limit)}

@app.get("/events")
async def events_endpoint():
    return {"events": get_all_events()}

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
