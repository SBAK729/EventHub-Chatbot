from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime, timedelta
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# -----------------------------
# Initialize model and database
# -----------------------------
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

try:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="event_content")
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")
    raise

# -----------------------------
# Sample events
# -----------------------------
events = [
    {
        "id": "1",
        "title": "Tech Conference 2023",
        "description": "Annual technology conference featuring talks on AI, machine learning, and software development. Join industry leaders for three days of innovation.",
        "category": "Technology",
        "date": "2023-11-15",
        "location": "San Francisco, CA",
        "tags": ["technology", "AI", "machine learning", "software", "conference"]
    },
    {
        "id": "2",
        "title": "Jazz Festival",
        "description": "Weekend jazz festival with performances from world-renowned artists. Food and drinks available throughout the event.",
        "category": "Music",
        "date": "2023-09-22",
        "location": "New York, NY",
        "tags": ["music", "jazz", "festival", "live performance", "food"]
    },
    {
        "id": "3",
        "title": "Startup Pitch Competition",
        "description": "Watch promising startups pitch their ideas to a panel of investors. Great opportunity for networking with entrepreneurs.",
        "category": "Business",
        "date": "2023-10-05",
        "location": "Austin, TX",
        "tags": ["business", "startup", "pitching", "investors", "networking"]
    },
    {
        "id": "4",
        "title": "Marathon for Charity",
        "description": "Annual marathon raising funds for children's hospitals. All fitness levels welcome with 5K, 10K, and full marathon options.",
        "category": "Sports",
        "date": "2023-10-28",
        "location": "Chicago, IL",
        "tags": ["sports", "marathon", "charity", "fitness", "running"]
    },
    {
        "id": "5",
        "title": "Digital Marketing Workshop",
        "description": "Hands-on workshop teaching the latest digital marketing strategies, SEO techniques, and social media advertising.",
        "category": "Education",
        "date": "2023-09-30",
        "location": "Online Event",
        "tags": ["education", "marketing", "SEO", "social media", "workshop"]
    },
    {
        "id": "6",
        "title": "Food & Wine Festival",
        "description": "Sample gourmet foods and fine wines from top chefs and vineyards. Cooking demonstrations and tasting sessions throughout the day.",
        "category": "Food & Drink",
        "date": "2023-11-10",
        "location": "Napa Valley, CA",
        "tags": ["food", "wine", "festival", "cooking", "tasting"]
    },
    {
        "id": "7",
        "title": "VR Gaming Expo",
        "description": "Experience the latest in virtual reality gaming technology. Try new games, meet developers, and participate in tournaments.",
        "category": "Gaming",
        "date": "2023-10-14",
        "location": "Los Angeles, CA",
        "tags": ["gaming", "VR", "virtual reality", "expo", "tournaments"]
    },
    {
        "id": "8",
        "title": "Yoga & Wellness Retreat",
        "description": "Weekend retreat focusing on yoga, meditation, and holistic wellness practices. Suitable for all experience levels.",
        "category": "Health & Wellness",
        "date": "2023-11-05",
        "location": "Sedona, AZ",
        "tags": ["health", "wellness", "yoga", "meditation", "retreat"]
    },
    {
        "id": "9",
        "title": "Tech Meetup: AI Applications",
        "description": "Monthly meetup discussing practical applications of artificial intelligence in various industries. Networking session included.",
        "category": "Technology",
        "date": "2023-10-12",
        "location": "San Francisco, CA",
        "tags": ["technology", "AI", "meetup", "networking", "applications"]
    },
    {
        "id": "10",
        "title": "Blockchain Conference",
        "description": "Exploring the future of blockchain technology, cryptocurrencies, and decentralized applications. Features industry experts.",
        "category": "Technology",
        "date": "2023-11-20",
        "location": "Austin, TX",
        "tags": ["technology", "blockchain", "crypto", "conference", "decentralized"]
    }
]

# -----------------------------
# Prepare embeddings
# -----------------------------
documents = []
metadatas = []
ids = []

for event in events:
    search_text = f"{event['title']}. {event['description']}. Category: {event['category']}. Location: {event['location']}. Date: {event['date']}. Tags: {', '.join(event['tags'])}"
    documents.append(search_text)
    metadatas.append({
        "title": event["title"],
        "description": event["description"],
        "category": event["category"],
        "date": event["date"],
        "location": event["location"],
        "tags": ", ".join(event["tags"])  
    })
    ids.append(event["id"])

try:
    embeddings = model.encode(documents).tolist()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    logger.info("Embeddings generated and added to ChromaDB successfully")
except Exception as e:
    logger.error(f"Error generating embeddings: {e}")
    raise

# -----------------------------
# Search history
# -----------------------------
search_history: List[Dict] = []

# -----------------------------
# Helper: extract filters
# -----------------------------
def extract_filters_from_query(query: str):
    filters = {}
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

    locations = ["san francisco", "new york", "austin", "chicago", "los angeles", 
                 "napa valley", "sedona", "online"]
    for location in locations:
        if location in query.lower():
            filters["location"] = location.title()
            query = re.sub(location, "", query, flags=re.IGNORECASE)

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
def perform_search(query: str, n_results: int = 10):
    try:
        # Store history
        search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat()
        })

        # Extract filters
        processed_query, filters = extract_filters_from_query(query)

        # Generate query embedding
        query_embedding = model.encode([processed_query]).tolist()

        # Prepare ChromaDB filter
        where_filter = {k: v for k, v in filters.items()} if filters else None

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter
        )

        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                # convert tags back to array
                tags = [tag.strip() for tag in metadata.get("tags", "").split(",")]
                metadata["tags"] = tags
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": metadata,
                    "distance": results['distances'][0][i]
                })

        return {
            "results": formatted_results,
            "processed_query": processed_query,
            "filters_applied": filters
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise

def get_search_history(limit: int = 10):
    return search_history[-limit:]

def get_all_events():
    return events
