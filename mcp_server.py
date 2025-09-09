"""
Simplified MCP Server for EventHub RAG functionality
Uses basic JSON-RPC over stdio instead of complex MCP libraries
"""

import asyncio
import json
import sys
import os
from typing import Any, Dict

sys.path.append('.')
from components.search.search import perform_search, get_all_events, initialize_embeddings

class SimpleMCPServer:
    def __init__(self):
        self.tools = {
            "search_events": {
                "name": "search_events",
                "description": "Search for events using natural language query with filters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query for events"},
                        "user_id": {"type": "string", "description": "User identifier", "default": "default"}
                    },
                    "required": ["query"]
                }
            },
            "get_all_events": {
                "name": "get_all_events", 
                "description": "Get all available events without filtering",
                "inputSchema": {"type": "object", "properties": {}}
            }
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC request"""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "tools/list":
                result = {"tools": list(self.tools.values())}
            
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                if tool_name == "search_events":
                    result = await self.search_events(tool_args)
                elif tool_name == "get_all_events":
                    result = await self.get_all_events(tool_args)
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0", 
                "id": request.get("id"),
                "error": {"code": -1, "message": str(e)}
            }
    
    async def search_events(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_events tool call"""
        query = args.get("query", "")
        user_id = args.get("user_id", "default")
        
        try:
            results = perform_search(query, user_id)
            formatted_results = []
            for event in results:
                formatted_event = {
                    "title": event.get("title", ""),
                    "category": event.get("category", ""),
                    "location": event.get("location", ""),
                    "startDateTime": event.get("startDateTime", ""),
                    "endDateTime": event.get("endDateTime", ""),
                    "price": event.get("price", ""),
                    "isFree": event.get("isFree", False),
                    "organizer": event.get("organizer", ""),
                    "tags": event.get("tags", "")
                }
                formatted_results.append(formatted_event)
            
            response_data = {
                "query": query,
                "results_count": len(formatted_results),
                "events": formatted_results[:10]
            }
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response_data, indent=2)
                }]
            }
            
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error searching events: {str(e)}"
                }]
            }
    
    async def get_all_events(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_all_events tool call"""
        try:
            events = get_all_events()
            response_data = {
                "total_events": len(events),
                "events": events[:20]
            }            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(response_data, indent=2)
                }]
            }            
        except Exception as e:
            return {
                "content": [{
                    "type": "text", 
                    "text": f"Error getting events: {str(e)}"
                }]
            }
    
    async def run(self):
        """Main server loop"""
        print("✓ Simple MCP Server started", file=sys.stderr)
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break              
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -1, "message": str(e)}
                }
                print(json.dumps(error_response))
                sys.stdout.flush()

async def main():
    """Run the simplified MCP server"""
    try:
        initialize_embeddings()
        print("✓ Event embeddings initialized", file=sys.stderr)
    except Exception as e:
        print(f"✗ Failed to initialize embeddings: {e}", file=sys.stderr)
    server = SimpleMCPServer()
    await server.run()