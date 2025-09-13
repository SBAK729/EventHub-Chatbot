    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    import logging
    import threading
    import time
    import asyncio
    from typing import Dict, Optional
    import os
    # Import core components
    from components.search.search import perform_search, get_all_events, initialize_embeddings, refresh_embeddings
    from components.Event_ai.crew import EventContentCrew

    # Import MCP server
    from mcp_server import SimpleMCPServer

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
        allow_origins=["https://event-hub0.vercel.app/","https://eventhu.vercel.app/","http://localhost:3000/"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _mcp_server_instance: Optional[SimpleMCPServer] = None
    _mcp_server_started = False

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
    _mcp_server_instance: Optional[SimpleMCPServer] = None

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
        global _mcp_server_started, _mcp_server_instance
        
        logger.info("→ Starting EventHub application")
        
        # Initialize embeddings asynchronously
        try:
            logger.info("→ Initializing embeddings...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, initialize_embeddings)
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

        # Launch MCP server instance
        try:
            if not _mcp_server_started:
                logger.info("→ Starting MCP server instance...")
                _mcp_server_instance = SimpleMCPServer()
                _mcp_server_started = True
                logger.info("✓ MCP server instance ready")
        except Exception as e:
            logger.error(f"✗ Failed to start MCP server: {e}")

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

            # Access tasks_output safely
            tasks = getattr(result, "tasks_output", [])
            if tasks:
                last_task = tasks[-1]
                # Use vars() to convert TaskOutput object to dict
                last_task_dict = vars(last_task) if hasattr(last_task, "__dict__") else {}
                proposals = last_task_dict.get("json_dict", {}).get("proposals", [])
            else:
                proposals = getattr(result, "proposals", [])

            return {"proposals": proposals}

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
    # MCP HTTP Bridge Routes
    # -----------------------------
    @app.post("/mcp/tools/call")
    async def mcp_tools_call(request: Dict):
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

    if __name__ == "__main__":
        logger.info("→ Starting EventHub server on http://localhost:8000")
        logger.info("→ API documentation available at http://localhost:8000/docs")
        
        try:
            port = int(os.getenv("PORT", 8000))
            uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
        except Exception as e:
            logger.error(f"✗ Failed to start server: {e}")
            raise

