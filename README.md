# SHL Assessment Recommender System

A production-ready conversational AI recommender for SHL assessments. Built on FastAPI and a hybrid RAG (FAISS + TF-IDF) retrieval pipeline, it grounds recommendation shortlists directly in the official SHL product catalog.

---

## ✨ Features

- **Stateless Turn Management**: Adheres strictly to the stateless multi-turn evaluator specification (every request carries the full `messages` history).
- **Hybrid RAG Pipeline**: Combines dense semantic retrieval (`all-MiniLM-L6-v2` + `FAISS`) with sparse keyword matching (`TF-IDF`) to maximize **Recall@10** and match exact skills.
- **Smart Offline Fallback Mode**: Run the application out-of-the-box! If no API keys (`GEMINI_API_KEY` or `OPENAI_API_KEY`) are present, a rule-based mock engine automatically extracts role requirements, queries the index, and serves accurate recommendations.
- **Dual LLM Provider Support**: Fully supports Google Gemini (via new `google-genai` SDK or legacy fallback) and OpenAI-compatible endpoints (e.g. OpenAI, Groq, OpenRouter, Ollama).
- **Evaluation Guardrails**: Defends against prompt injection, blocks off-topic queries (e.g. pizza, weather), ensures URL integrity, and caps conversation length at 8 turns to fit timeout limits.
- **Premium Frontend Dashboard**: Interactive dark-mode glassmorphism chat UI with a live shortlist sidebar, configuration options, and start-up chips.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **LLM SDKs**: `google-genai` & `httpx` (OpenAI completion async client)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (running locally on CPU or MPS)
- **Vector Search**: `FAISS`
- **Keyword Search**: `scikit-learn` (TF-IDF Vectorizer)
- **Testing**: `pytest` & `anyio`

---

## 🏗️ Project Structure

```text
SHL_assignment/
├── app/                      # Main application source
│   ├── api/                  # API routing (GET /health, POST /chat)
│   ├── core/                 # Configuration, logging setup
│   ├── models/               # Pydantic request/response schemas
│   ├── services/             # Core engines
│   │   ├── llm/              # LLM wrapper & prompts
│   │   ├── retrieval/        # FAISS vector store, TF-IDF searcher, hybrid RRF
│   │   ├── orchestration/    # Turn cap analyzer, comparison & refusal engines
│   │   └── evaluation/       # Response & URL validators
│   └── utils/                # Simple caching layers
├── data/                     # Source data
│   └── catalog_seed.json     # 74 scraped SHL assessments
│   └── vectorstore/          # Cached FAISS index and TF-IDF pickle
├── frontend/                 # Static web dashboard files
│   └── index.html            # Premium glassmorphism chat interface
├── scripts/                  # Evaluation & benchmark scripts
│   └── evaluator_tests.py    # 4-suite automated evaluation harness
├── tests/                    # Pytest test suite
│   └── test_api.py           # Async API tests
├── Dockerfile                # Production Docker configuration
├── render.yaml               # Render infrastructure template
└── requirements.txt          # Package dependencies
```

---

## 🚀 Getting Started

### 1. Setup Environment
Clone the repository, create a virtual environment, and install dependencies:
```bash
git clone https://github.com/lakshya3004/SHL.git
cd SHL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Keys (Optional)
Copy the environment template:
```bash
cp .env.example .env
```
Open `.env` and fill in your keys:
- For **Gemini**: Set `GEMINI_API_KEY`.
- For **OpenAI / Groq / OpenRouter**: Leave `GEMINI_API_KEY` blank, set `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`, and your endpoint's `LLM_BASE_URL`.
- *Note: If both are left blank or contain placeholder text, the service automatically runs in local mock fallback mode.*

### 3. Run the Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
On first startup, the service self-initializes: downloads the embedding model and builds FAISS+TF-IDF indexes from `data/catalog_seed.json` (~12 seconds). Subsequent start-ups load indices directly from disk in under a second.

### 4. Access the Dashboard
Open your browser and navigate to **[http://localhost:8000](http://localhost:8000)** to interact with the conversational dashboard.

---

## 📡 API Specifications

The evaluator runs at root-level endpoints and requires strict schema structures.

### 1. Health check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "ok", "version": "1.0.0", ...}`

### 2. Chat completion
- **Endpoint**: `POST /chat`
- **Request Body**:
  ```json
  {
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer"},
      {"role": "assistant", "content": "What is the seniority level?"},
      {"role": "user", "content": "Mid-level, around 4 years of experience"}
    ]
  }
  ```
- **Response Body**:
  ```json
  {
    "reply": "Based on your requirements, I recommend these Java assessments...",
    "recommendations": [
      {
        "name": "Java Programming",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/java/",
        "test_type": "K"
      }
    ],
    "end_of_conversation": false
  }
  ```

---

## 🧪 Evaluation & Testing

### Run Pytest Suite
Verifies API response status codes, root paths, and strict schema compliance:
```bash
PYTHONPATH=. pytest tests/
```

### Run Automated Evaluation Harness
Runs the 4 test suites simulating health checks, 10 behavioral probes (refusals, vague queries, schema validations), Recall@10 metrics across traces, and quality checks:
```bash
python3 scripts/evaluator_tests.py --url http://localhost:8000
```
