# SHL Assessment Recommender: Technical Approach

This document outlines the design decisions, retrieval architecture, conversational orchestration logic, and evaluation strategies implemented for the SHL Assessment Recommender take-home assignment.

---

## 🏗️ System Architecture

The system utilizes a modular, production-oriented **Retrieval-Augmented Generation (RAG)** architecture built on FastAPI. It separates concerns across clear abstraction boundaries:

```
[Client (UI / Evaluator)]
         │
         ▼
     [FastAPI] (POST /chat, GET /health)
         │
         ▼
[ChatOrchestrator]
   ├── [ConversationAnalyzer] (Intent detection: Recommend, Clarify, Compare, Refuse)
   ├── [RecommendationEngine] (Dynamic query synthesis, hybrid retrieval trigger)
   │        ├── [FAISS Index] (Dense semantic vector similarity)
   │        └── [TF-IDF Index] (Sparse token matching)
   ├── [ComparisonEngine]     (Assessment catalog difference generator)
   ├── [RefusalEngine]        (Scope gatekeeper & prompt injection shield)
   ├── [ContextBuilder]       (Catalog context markdown formatting)
   ├── [ResponseGenerator]    (LLM / Fallback text response generation)
   └── [Validators]
            ├── [RetrievalValidator] (Deduplication, 1-10 count limit)
            └── [ResponseValidator]  (URL grounding check, final EOC check)
```

---

## 🔍 Hybrid Retrieval Design & Recall@10

Through baseline benchmarking against user persona traces, we discovered that dense semantic search alone (using vector embeddings) struggled with recruiter queries containing exact technology names (e.g. "Spring Boot") or specific assessment nomenclature. 

To achieve a high **Recall@10** score, we implemented a **Hybrid Retrieval** pipeline:
1. **Dense Vector Search**: Embeds queries using `sentence-transformers/all-MiniLM-L6-v2` and searches a `FAISS` index using inner product (cosine similarity) to capture user intent and role analogies.
2. **Sparse Keyword Search**: Uses a `TF-IDF` vectorizer fitting unigrams and bigrams to capture exact programming languages, frameworks, or assessment acronyms (e.g., "OPQ", "Verify").
3. **Reciprocal Rank Fusion (RRF)**: Merges the ranked outputs of both retrievers using the standard RRF algorithm:
   $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
   This approach dynamically prioritizes candidates that rank highly across both semantic context and keyword matching.

---

## 🤖 Conversational Orchestration & State Machine

The recommender is completely stateless; every transaction parses the full `messages` history payload to maintain session continuity.

### 1. Intent Analysis
The `ConversationAnalyzer` classifies the conversation state into one of four distinct modes:
* **`clarify`**: Triggered when queries are vague (e.g., "I need an assessment"). The agent responds with clarifying questions without rendering premature recommendations.
* **`recommend`**: Activated when role, skills, or seniority constraints are sufficient. The agent queries the hybrid index and returns 1 to 10 assessments.
* **`compare`**: Triggered when comparison between tests is requested. The `ComparisonEngine` provides grounded comparisons drawn strictly from retrieved catalog facts.
* **`refuse`**: Gatekeeps out-of-scope requests, salary discussions, legal advice, or prompt-injection jailbreak attempts.

### 2. Safeguards & End-of-Conversation (EOC)
* **Turn Cap Enforcement**: If the conversation reaches 4+ user turns, the analyzer automatically flags the state as `is_sufficient = True` and forces recommendations to ensure the conversation wraps up within the evaluator's 8-turn hard cap.
* **Intelligent EOC State Machine**: Rather than ending the conversation immediately after Turn 2, the orchestrator only sets `end_of_conversation: true` if:
  1. The user has had at least 2 turns **AND** the latest message contains confirmation/acceptance keywords (e.g. *perfect*, *confirmed*, *thanks*, *looks good*).
  2. The turn count has reached the safety threshold (>= 4 turns) to prevent timeouts.
  This allows human testers to ask refinement questions without seeing premature checkmark closure alerts on the UI.

---

## 🔌 Smart Offline Fallback Mode (Key Innovation)

To ensure the application is **fully functional out-of-the-box** without initial API key configuration:
* The `LLMService` automatically identifies if the provided API keys (`GEMINI_API_KEY` or `OPENAI_API_KEY`) are missing, empty, or contain default placeholder text (e.g., `your_gemini_api_key_here`).
* In the absence of a active key, it seamlessly falls back to a **Deterministic Mock Fallback Mode**.
* This mock mode parses query history keywords, extracts role requirements, queries the live FAISS/TF-IDF database, and returns real, valid catalog recommendations and grounded replies.
* As soon as valid keys are provided in `.env`, the service instantly hot-reloads and routes all queries through the active LLM.

---

## ⚙️ Production Deployment Optimizations (Render)

Deploying large ML models on free-tier container platforms like Render often triggers cold-start timeouts. We solved this with two major Docker build optimizations:
1. **Pre-Cached Models**: The `sentence-transformers` model weights are downloaded and stored inside the image during the Docker `build` phase.
2. **Pre-Built Database**: Retrieval indices (FAISS and TF-IDF pickles) are built directly during the Docker `build` phase using `data/catalog_seed.json`.

**Result**: Container startup on Render is completely instant (< 1 second) and has zero online model download dependencies, fully resolving deployment timeout issues.
