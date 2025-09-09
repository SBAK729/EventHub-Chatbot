import logging
from datetime import datetime, timedelta
import re
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
from events_data import events

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
PERSISTENT_DIRECTORY = "./chroma_db"
EVENTS_API_URL = "https://your-events-api.com/events"  # Replace with your actual API
EMBEDDINGS_CACHE_FILE = "./embeddings_cache.json"

# -----------------------------
# Global collection variable
# -----------------------------
collection = None

# -----------------------------
# Initialize model
# -----------------------------
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# -----------------------------
# Initialize ChromaDB with persistent storage
# -----------------------------
def get_chroma_client():
    """Get ChromaDB client with persistent storage"""
    return chromadb.Client(Settings(
        persist_directory=PERSISTENT_DIRECTORY
    ))

def initialize_collection():
    """Initialize or get existing collection"""
    client = get_chroma_client()
    try:
        collection = client.get_collection(name="event_content")
        logger.info("Using existing ChromaDB collection")
    except:
        collection = client.create_collection(name="event_content")
        logger.info("Created new ChromaDB collection")
    return collection

# -----------------------------
# Sample events (fallback)
# -----------------------------
sample_events = events

# -----------------------------
# Embedding management functions
# -----------------------------
def initialize_embeddings():
    """Initialize embeddings on application startup"""
    global collection
    collection = initialize_collection()
    
    # Check if collection is empty and needs initial population
    if collection.count() == 0:
        logger.info("Collection empty, generating initial embeddings...")
        refresh_embeddings()
    else:
        logger.info(f"Collection already has {collection.count()} embeddings")

def refresh_embeddings():
    """Refresh embeddings by recreating the collection"""
    global collection
    
    # Get current events (all events, no user filtering at storage level)
    events = fetch_events()
    
    # Recreate the collection
    try:
        client = get_chroma_client()
        try:
            client.delete_collection(name="event_content")
            logger.info("Deleted existing collection")
        except:
            logger.info("No existing collection to delete")
        
        collection = client.create_collection(name="event_content")
        logger.info("Created new collection")
        
    except Exception as e:
        logger.error(f"Error recreating collection: {e}")
        raise
    
    # Generate new embeddings (store all events with user_id as "global")
    documents = []
    metadatas = []
    ids = []

    for event in events:
        search_text = generate_search_text(event)
        documents.append(search_text)

        # Store all events with user_id as "global" - filtering happens during search
        metadatas.append({
            "user_id": "global",  # All events stored as global
            "_id": str(event["_id"]),
            "title": event.get("title", ""),
            "location": event.get("location") or "",
            "createdAt": event.get("createdAt") or "",
            "imageUrl": event.get("imageUrl") or "",
            "startDateTime": event.get("startDateTime") or "",
            "endDateTime": event.get("endDateTime") or "",
            "price": event.get("price") or "",
            "isFree": event.get("isFree", False),
            "category": event.get("category", {}).get("name", ""),
            "organizer": f"{event.get('organizer', {}).get('firstName','')} {event.get('organizer', {}).get('lastName','')}".strip(),
            "tags": ", ".join(event.get("tags", []))
        })

        ids.append(f"global_{event['_id']}")  

    try:
        # Process in batches to manage memory
        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            batch_embeddings = model.encode(batch_docs).tolist()
            
            collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        logger.info(f"Embeddings refreshed successfully for {len(events)} events")
        
    except Exception as e:
        logger.error(f"Error refreshing embeddings: {e}")
        raise

def generate_search_text(event):
    """Generate search text from event data"""
    return (
        f"Title: {event['title']}. "
        f"Description: {event.get('description', '')}. "
        f"Category: {event.get('category', {}).get('name', '')}. "
        f"Location: {event.get('location', '')}. "
        f"Tags: {', '.join(event.get('tags', []))}. "
        f"Organizer: {event.get('organizer', {}).get('firstName', '')} {event.get('organizer', {}).get('lastName', '')}. "
        f"Start: {event.get('startDateTime', '')}. "
        f"End: {event.get('endDateTime', '')}. "
        f"Price: {event.get('price', '')}. "
        f"Free: {'Yes' if event.get('isFree', False) else 'No'}. "
        f"URL: {event.get('url', '')}"
    )

def fetch_events():
    # """Fetch events from API with fallback to sample data"""
    # try:
    #     response = requests.get(EVENTS_API_URL, timeout=10)
    #     response.raise_for_status()
    #     events = response.json()
    #     logger.info(f"Fetched {len(events)} events from API")
    #     return events
    # except Exception as e:
    #     logger.warning(f"Error fetching events from API, using sample data: {e}")
    return sample_events

# -----------------------------
# Helper: extract filters
# -----------------------------
def extract_filters_from_query(query: str):
    filters = {}

    # Free / Paid
    if "free" in query.lower():
        filters["isFree"] = True
        query = re.sub(r"\bfree\b", "", query, flags=re.IGNORECASE)
    if "paid" in query.lower():
        filters["isFree"] = False
        query = re.sub(r"\bpaid\b", "", query, flags=re.IGNORECASE)

    # Location detection
    location_match = re.search(r"in\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    if location_match:
        filters["location"] = location_match.group(1).strip().title()
        query = re.sub(location_match.group(0), "", query, flags=re.IGNORECASE)

    # Date detection
    today = datetime.now().date()
    date_patterns = {
        "today": today,
        "tomorrow": today + timedelta(days=1),
        "this weekend": today + timedelta(days=(5 - today.weekday()) % 7),
        "next week": today + timedelta(days=7),
    }
    for pattern, date_value in date_patterns.items():
        if pattern in query.lower():
            filters["date"] = date_value.strftime("%Y-%m-%d")
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)

    # Category detection
    categories = ["technology", "music", "business", "sports", "education",
                  "food & drink", "gaming", "health & wellness"]
    for category in categories:
        if category in query.lower():
            filters["category"] = category.title()
            query = re.sub(category, "", query, flags=re.IGNORECASE)

    query = re.sub(r'\s+', ' ', query).strip()
    return query, filters

# -----------------------------
# Main search function
# -----------------------------
def perform_search(query: str, user_id: str):
    n_results = 10
    similarity_threshold = 0.2 

    try:
        if collection is None:
            initialize_embeddings()

        processed_query, filters = extract_filters_from_query(query)

        # Base filter: only return global events
        where_filter = {"user_id": "global"}

        # Add other filters from query parsing
        other_filters = []
        if "isFree" in filters:
            other_filters.append({"isFree": filters["isFree"]})
        if "location" in filters:
            other_filters.append({"location": filters["location"]})
        if "category" in filters:
            other_filters.append({"category": filters["category"]})

        if other_filters:
            where_filter = {
                "$and": [
                    {"user_id": "global"},
                    *other_filters
                ]
            }

        # Generate query embedding
        query_embedding = model.encode([processed_query]).tolist()[0]

        # Perform the search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=['metadatas', 'distances']
        )

        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not metadatas:
            return []

        # Convert distances to similarity scores
        scored_results = []
        for idx, metadata in enumerate(metadatas):
            distance = distances[idx]
            similarity_score = 1 / (1 + distance) if distance > 0 else 1.0

            # ✅ filter by threshold
            if similarity_score >= similarity_threshold:
                scored_results.append({
                    "metadata": metadata,
                    "score": similarity_score
                })

        # If nothing meets threshold → return empty
        if not scored_results:
            return []

        # Sort and return
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return [r["metadata"] for r in scored_results]

    except Exception as e:
        logger.error(f"Search error for user {user_id}: {e}")
        raise

# -----------------------------
# Return all events
# -----------------------------
def get_all_events():
    return fetch_events()