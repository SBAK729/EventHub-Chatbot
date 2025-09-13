# EventHub AI API

**EventHub AI API** provides semantic event search, AI-powered event proposal generation, and MCP server integration.
It leverages **FastAPI**, **ChromaDB**, and **CrewAI** to enable smart event discovery, embedding-based search, and automated event creation workflows.

---

## 🚀 Features

* **Semantic Event Search**: Find events using natural language queries (powered by `SentenceTransformer` + `ChromaDB`).
* **Event Proposal Generation**: AI-generated and validated event ideas using **CrewAI**.
* **Embeddings Refresh**: Automatic (every 30 minutes) and manual embedding refresh.
* **MCP Server Integration**: Exposes tools and requests through a Model Control Protocol (MCP) bridge.
* **API Documentation**: Built-in Swagger UI at `/docs`.

---

## 🛠️ Tech Stack

* **Backend**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
* **Database/Vector Store**: [ChromaDB](https://www.trychroma.com/)
* **Embeddings**: [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
* **Event AI Crew**: [CrewAI](https://github.com/joaomdmoura/crewAI) with **SerperDevTool** + **Gemini LLM** +**Validator Agent**
* **Task Scheduling**: Background threads for periodic refresh
* **Logging**: Python `logging` module

---

## 📂 Project Structure

```
.
├── app.py                        # Main FastAPI app
├── components/
│   ├── search/
│   │   └── search.py             # Semantic search, embeddings, ChromaDB
│   └── Event_ai/
│       └── crew.py               # EventContentCrew (CrewAI agents & tasks)
├── mcp_server.py                 # MCP server integration
├── config/
│   ├── agents.yaml               # Agent configurations
│   └── tasks.yaml                # Task configurations
├── events_data.py                # Sample/fallback event data
└── requirements.txt              # Dependencies
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/SBAK729/EventHub-Chatbot.git
cd EventHub-Chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file:

```env
# API Keys
SERPER_API_KEY=your_serper_api_key
GEMINI_API_KEY=your_gemini_api_key

# Server
PORT=8000
```

---

## ▶️ Running the API

### Development Mode (hot reload)

```bash
uvicorn app:app --reload
```


The server will start at:
👉 `http://localhost:8000`
👉 Swagger UI: `http://localhost:8000/docs`

---

## 📌 API Endpoints

### **Health Check**

`GET /`
Returns API status.

---

### **Search Events**

`POST /search`

**Request**:

```json
{
  "query": "free music events in New York",
  "user_id": "user123"
}
```

**Response**:

```json
{
  "results": [
    {
      "_id": "123",
      "title": "Jazz Night NYC",
      "location": "New York",
      "isFree": true,
      "category": "Music",
      "startDateTime": "2025-09-20T19:00:00"
    }
  ]
}
```

---

### **Generate Event Proposals**

`POST /generate-event`

**Request**:

```json
{
  "description": "A tech networking meetup with AI experts",
  "title": "AI Connect 2025"
}
```

**Response**:

```json
{
  "proposals": [
    {
      "title": "AI Innovation Meetup",
      "description": "An evening for AI enthusiasts and professionals...",
      "tags": ["AI", "Networking", "Technology", "Innovation", "Startup"]
    }
  ]
}
```

---

### **Manual Embeddings Refresh**

`POST /refresh-embeddings`

Forces embeddings regeneration.

---

### **MCP Bridge**

* `POST /mcp/tools/call` → Call a tool via MCP
* `POST /mcp/tools/list` → List available tools

---

## 🔄 Embeddings Lifecycle

* **Startup**: `initialize_embeddings()` loads or creates ChromaDB collection.
* **Background Task**: Every 30 min → `refresh_embeddings()` runs.
* **Manual Trigger**: `POST /refresh-embeddings`.

---

## 📜 Logging

Logs are output in the format:

```
2025-09-13 12:00:00 - app - INFO - → Starting EventHub application
2025-09-13 12:00:01 - app - INFO - ✓ Embeddings initialized successfully
```

Example Dockerfile snippet:

```dockerfile
# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port (Render uses 10000+ dynamic ports, we'll override with CMD)
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

```

---

## 🧑‍💻 Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/awesome-feature`
3. Commit changes: `git commit -m "Add awesome feature"`
4. Push branch: `git push origin feature/awesome-feature`
5. Open PR 🚀

