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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
    user_id: str = "default"

class EventInput(BaseModel):
    description: str
    title: str

# -----------------------------
# Global state tracking
# -----------------------------
_mcp_server_started = False

# -----------------------------
# Background task for periodic refresh
# -----------------------------
def periodic_refresh():
    """Refresh embeddings every 30 minutes"""
    logger.info("→ Starting periodic refresh thread")
    
    while True:
        try:
            time.sleep(1800)  # 30 minutes
            logger.info("→ Starting periodic embedding refresh...")
            refresh_embeddings()
            logger.info("✓ Embeddings refreshed successfully")
        except Exception as e:
            logger.error(f"✗ Error during periodic refresh: {e}")

# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global _mcp_server_started
    
    logger.info("→ Starting EventHub application")
    
    # Initialize embeddings
    try:
        logger.info("→ Initializing embeddings...")
        initialize_embeddings()
        logger.info("✓ Embeddings initialized successfully")
    except Exception as e:
        logger.error(f"✗ Error initializing embeddings: {e}")
        logger.warning("→ Continuing without embeddings (search functionality may be limited)")

    # Start background refresh thread
    try:
        refresh_thread = threading.Thread(target=periodic_refresh, daemon=True)
        refresh_thread.start()
        logger.info("✓ Background refresh thread started")
    except Exception as e:
        logger.error(f"✗ Failed to start refresh thread: {e}")

    # Launch MCP server in background
    try:
        if not _mcp_server_started:
            logger.info("→ Starting MCP server in background...")
            asyncio.create_task(mcp_main())
            _mcp_server_started = True
            logger.info("✓ MCP server started successfully")
    except Exception as e:
        logger.error(f"✗ Failed to start MCP server: {e}")

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def health_check():
    logger.info("→ Health check requested")
    return {
        "status": "healthy", 
        "message": "EventHub API is running",
        "mcp_server_active": _mcp_server_started
    }

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    logger.info(f"→ Search request: query='{request.query}', user_id='{request.user_id}'")
    
    try:
        results = perform_search(request.query, request.user_id)
        result_count = len(results) if results else 0
        
        logger.info(f"✓ Search completed: found {result_count} results")
        
        if not results:
            logger.info("→ No matching events found")
            return {"results": [], "message": "No matching events found", "count": 0}
        
        return {
            "results": results,
            "count": result_count,
            "query": request.query
        }
        
    except Exception as e:
        logger.error(f"✗ Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/generate-event")
def generate_event(event: EventInput):
    logger.info(f"→ Event generation request: title='{event.title}'")
    
    try:
        crew = EventContentCrew()
        logger.info("→ Starting event content generation crew...")
        
        result = crew.eventcrew().kickoff(
            inputs={"event_description": event.description, "title": event.title}
        )
        
        proposals = result.get("proposals", [])
        proposal_count = len(proposals) if proposals else 0
        
        logger.info(f"✓ Event generation completed: {proposal_count} proposals created")
        
        return {
            "proposals": proposals,
            "count": proposal_count,
            "title": event.title
        }
        
    except Exception as e:
        logger.error(f"✗ Generate-event error: {e}")
        raise HTTPException(status_code=500, detail=f"Event generation failed: {str(e)}")

@app.post("/refresh-embeddings")
async def manual_refresh():
    logger.info("→ Manual embedding refresh requested")
    
    try:
        refresh_embeddings()
        logger.info("✓ Manual embedding refresh completed")
        return {
            "status": "success", 
            "message": "Embeddings refreshed successfully"
        }
    except Exception as e:
        logger.error(f"✗ Manual refresh error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding refresh failed: {str(e)}")

@app.get("/events")
async def get_events_endpoint():
    """Get all events endpoint for testing"""
    logger.info("→ Get all events request")
    
    try:
        events = get_all_events()
        event_count = len(events) if events else 0
        
        logger.info(f"✓ Retrieved {event_count} total events")
        
        return {
            "events": events[:50],  # Limit for API response
            "total_count": event_count,
            "returned_count": min(50, event_count)
        }
        
    except Exception as e:
        logger.error(f"✗ Get events error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve events: {str(e)}")

# -----------------------------
# MCP HTTP Bridge Routes
# -----------------------------
@app.post("/mcp/tools/call")
async def mcp_tools_call(request: Dict):
    """HTTP bridge for MCP tools/call"""
    global _mcp_server_instance
    
    if not _mcp_server_instance:
        logger.error("✗ MCP server instance not available")
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    logger.info(f"→ MCP bridge request: {request.get('method', 'unknown')}")
    
    try:
        response = await _mcp_server_instance.handle_request(request)
        logger.info("✓ MCP bridge request completed")
        return response
    except Exception as e:
        logger.error(f"✗ MCP bridge error: {e}")
        raise HTTPException(status_code=500, detail=f"MCP request failed: {str(e)}")

@app.post("/mcp/tools/list") 
async def mcp_tools_list():
    """HTTP bridge for MCP tools/list"""
    global _mcp_server_instance
    
    if not _mcp_server_instance:
        logger.error("✗ MCP server instance not available")
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    
    logger.info("→ MCP tools list request")
    
    try:
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        response = await _mcp_server_instance.handle_request(request)
        logger.info("✓ MCP tools list completed")
        return response
    except Exception as e:
        logger.error(f"✗ MCP tools list error: {e}")
        raise HTTPException(status_code=500, detail=f"MCP tools list failed: {str(e)}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    logger.info("→ Starting EventHub server on http://localhost:8000")
    logger.info("→ API documentation available at http://localhost:8000/docs")
    
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"✗ Failed to start server: {e}")
        raise