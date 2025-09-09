# EventHub-Chatbot
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