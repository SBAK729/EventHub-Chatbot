
import logging
from datetime import datetime, timedelta
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger(__name__)

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
# Initialize ChromaDB
# -----------------------------
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
        "_id": "1",
        "title": "Tech Conference 2023",
        "description": "Annual technology conference featuring talks on AI, machine learning, and software development. Join industry leaders for three days of innovation.",
        "tags": ["technology", "AI", "machine learning", "software", "conference"],
        "location": "San Francisco, CA",
        "createdAt": "2023-08-01T00:00:00",
        "imageUrl": "https://example.com/event1.jpg",
        "startDateTime": "2023-11-15T09:00:00",
        "endDateTime": "2023-11-15T17:00:00",
        "price": "TBD",
        "isFree": False,
        "url": "https://example.com/events/1",
        "category": {"_id": "c1", "name": "Technology"},
        "organizer": {"_id": "o1", "firstName": "John", "lastName": "Doe"}
    },
    {
        "_id": "2",
        "title": "Jazz Festival",
        "description": "Weekend jazz festival with performances from world-renowned artists. Food and drinks available throughout the event.",
        "tags": ["music", "jazz", "festival", "live performance", "food"],
        "location": "New York, NY",
        "createdAt": "2023-08-02T00:00:00",
        "imageUrl": "https://example.com/event2.jpg",
        "startDateTime": "2023-09-22T10:00:00",
        "endDateTime": "2023-09-22T22:00:00",
        "price": "50",
        "isFree": False,
        "url": "https://example.com/events/2",
        "category": {"_id": "c2", "name": "Music"},
        "organizer": {"_id": "o2", "firstName": "Alice", "lastName": "Smith"}
    },
    {
        "_id": "3",
        "title": "Startup Pitch Competition",
        "description": "Watch promising startups pitch their ideas to a panel of investors. Great opportunity for networking with entrepreneurs.",
        "tags": ["business", "startup", "pitching", "investors", "networking"],
        "location": "Austin, TX",
        "createdAt": "2023-08-03T00:00:00",
        "imageUrl": "https://example.com/event3.jpg",
        "startDateTime": "2023-10-05T14:00:00",
        "endDateTime": "2023-10-05T18:00:00",
        "price": "0",
        "isFree": True,
        "url": "https://example.com/events/3",
        "category": {"_id": "c3", "name": "Business"},
        "organizer": {"_id": "o3", "firstName": "Bob", "lastName": "Johnson"}
    },
    {
        "_id": "4",
        "title": "Marathon for Charity",
        "description": "Annual marathon raising funds for children's hospitals. All fitness levels welcome with 5K, 10K, and full marathon options.",
        "tags": ["sports", "marathon", "charity", "fitness", "running"],
        "location": "Chicago, IL",
        "createdAt": "2023-08-04T00:00:00",
        "imageUrl": "https://example.com/event4.jpg",
        "startDateTime": "2023-10-28T07:00:00",
        "endDateTime": "2023-10-28T13:00:00",
        "price": "25",
        "isFree": False,
        "url": "https://example.com/events/4",
        "category": {"_id": "c4", "name": "Sports"},
        "organizer": {"_id": "o4", "firstName": "Sarah", "lastName": "Lee"}
    },
    {
        "_id": "5",
        "title": "Digital Marketing Workshop",
        "description": "Hands-on workshop teaching the latest digital marketing strategies, SEO techniques, and social media advertising.",
        "tags": ["education", "marketing", "SEO", "social media", "workshop"],
        "location": "Online Event",
        "createdAt": "2023-08-05T00:00:00",
        "imageUrl": "https://example.com/event5.jpg",
        "startDateTime": "2023-09-30T09:00:00",
        "endDateTime": "2023-09-30T16:00:00",
        "price": "0",
        "isFree": True,
        "url": "https://example.com/events/5",
        "category": {"_id": "c5", "name": "Education"},
        "organizer": {"_id": "o5", "firstName": "David", "lastName": "Kim"}
    },
    {
        "_id": "6",
        "title": "Food & Wine Festival",
        "description": "Sample gourmet foods and fine wines from top chefs and vineyards. Cooking demonstrations and tasting sessions throughout the day.",
        "tags": ["food", "wine", "festival", "cooking", "tasting"],
        "location": "Napa Valley, CA",
        "createdAt": "2023-08-06T00:00:00",
        "imageUrl": "https://example.com/event6.jpg",
        "startDateTime": "2023-11-10T11:00:00",
        "endDateTime": "2023-11-10T20:00:00",
        "price": "75",
        "isFree": False,
        "url": "https://example.com/events/6",
        "category": {"_id": "c6", "name": "Food & Drink"},
        "organizer": {"_id": "o6", "firstName": "Emma", "lastName": "Williams"}
    },
    {
        "_id": "7",
        "title": "VR Gaming Expo",
        "description": "Experience the latest in virtual reality gaming technology. Try new games, meet developers, and participate in tournaments.",
        "tags": ["gaming", "VR", "virtual reality", "expo", "tournaments"],
        "location": "Los Angeles, CA",
        "createdAt": "2023-08-07T00:00:00",
        "imageUrl": "https://example.com/event7.jpg",
        "startDateTime": "2023-10-14T10:00:00",
        "endDateTime": "2023-10-14T18:00:00",
        "price": "30",
        "isFree": False,
        "url": "https://example.com/events/7",
        "category": {"_id": "c7", "name": "Gaming"},
        "organizer": {"_id": "o7", "firstName": "Liam", "lastName": "Brown"}
    },
    {
        "_id": "8",
        "title": "Yoga & Wellness Retreat",
        "description": "Weekend retreat focusing on yoga, meditation, and holistic wellness practices. Suitable for all experience levels.",
        "tags": ["health", "wellness", "yoga", "meditation", "retreat"],
        "location": "Sedona, AZ",
        "createdAt": "2023-08-08T00:00:00",
        "imageUrl": "https://example.com/event8.jpg",
        "startDateTime": "2023-11-05T08:00:00",
        "endDateTime": "2023-11-05T17:00:00",
        "price": "100",
        "isFree": False,
        "url": "https://example.com/events/8",
        "category": {"_id": "c8", "name": "Health & Wellness"},
        "organizer": {"_id": "o8", "firstName": "Sophia", "lastName": "Martinez"}
    },
    {
        "_id": "9",
        "title": "Tech Meetup: AI Applications",
        "description": "Monthly meetup discussing practical applications of artificial intelligence in various industries. Networking session included.",
        "tags": ["technology", "AI", "meetup", "networking", "applications"],
        "location": "San Francisco, CA",
        "createdAt": "2023-08-09T00:00:00",
        "imageUrl": "https://example.com/event9.jpg",
        "startDateTime": "2023-10-12T18:00:00",
        "endDateTime": "2023-10-12T21:00:00",
        "price": "0",
        "isFree": True,
        "url": "https://example.com/events/9",
        "category": {"_id": "c9", "name": "Technology"},
        "organizer": {"_id": "o9", "firstName": "Olivia", "lastName": "Davis"}
    },
    {
        "_id": "10",
        "title": "Blockchain Conference",
        "description": "Exploring the future of blockchain technology, cryptocurrencies, and decentralized applications. Features industry experts.",
        "tags": ["technology", "blockchain", "crypto", "conference", "decentralized"],
        "location": "Austin, TX",
        "createdAt": "2023-08-10T00:00:00",
        "imageUrl": "https://example.com/event10.jpg",
        "startDateTime": "2023-11-20T09:00:00",
        "endDateTime": "2023-11-20T17:00:00",
        "price": "120",
        "isFree": False,
        "url": "https://example.com/events/10",
        "category": {"_id": "c10", "name": "Technology"},
        "organizer": {"_id": "o10", "firstName": "James", "lastName": "Miller"}
    }
]

# -----------------------------
# Prepare embeddings
# -----------------------------
documents = []
metadatas = []
ids = []

for event in events:
    search_text = (
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
    documents.append(search_text)

    metadatas.append({
        "user_id": str(event.get("user_id", "global")),
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

    ids.append(f"{event.get('user_id', 'global')}_{event['_id']}")

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

    try:
        processed_query, filters = extract_filters_from_query(query)

        # Ensure user_id filtering (global + user-specific)
        user_or_filter = [{"user_id": "global"}, {"user_id": user_id}]

        # Add other filters
        other_filters = []
        if "isFree" in filters:
            other_filters.append({"isFree": filters["isFree"]})
        if "location" in filters:
            other_filters.append({"location": filters["location"]})
        if "category" in filters:
            other_filters.append({"category": filters["category"]})

        # Combine everything under a single $and
        where_filter = {
            "$and": [
                {"$or": user_or_filter},
                *other_filters  # unpack other filters
            ]
        } if other_filters else {"$or": user_or_filter}


        query_embedding = model.encode([processed_query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=['metadatas', 'documents', 'embeddings']
        )

        metadatas = results.get("metadatas", [[]])[0]
        embeddings_list = results.get("embeddings", [[]])[0]

        if not metadatas:
            return []

        def cosine_similarity(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scored_results = []
        for idx, metadata in enumerate(metadatas):
            score = cosine_similarity(query_embedding[0], embeddings_list[idx])
            scored_results.append({"metadata": metadata, "cosine_score": score})

        scored_results.sort(key=lambda x: x["cosine_score"], reverse=True)
        return [r["metadata"] for r in scored_results]

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise

# -----------------------------
# Return all events
# -----------------------------
def get_all_events():
    #Get event data from api

    # try:
    #     response = requests.get(EVENTS_API_URL)
    #     response.raise_for_status()
    #     events = response.json()
    #     logger.info(f"Fetched {len(events)} events from API")
        # return events
    # except Exception as e:
    #     logger.error(f"Error fetching events: {e}")
    #     events= []

    return events
