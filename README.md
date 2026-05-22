# 🤖 RAG + LiteLLM Gateway — Production POC

> A production-grade Retrieval-Augmented Generation (RAG) system built phase-by-phase,
> featuring LiteLLM as the unified AI gateway across OpenAI, Groq, Gemini, and HuggingFace.

<img width="1086" height="1250" alt="poc" src="https://github.com/user-attachments/assets/6d638f45-e558-4630-9dcd-00d96eaff419" />




## 📌 Project Overview

This project demonstrates how to build a **complete RAG pipeline** from scratch and
progressively enhance it with enterprise-grade features using **LiteLLM** as the
central LLM gateway.

| | |
|---|---|
| **Stack** | FastAPI · LiteLLM · ChromaDB · sentence-transformers · Redis |
| **LLM Providers** | OpenAI · Groq (free) · Gemini (free) · HuggingFace (free) |
| **Embeddings** | BAAI/bge-small-en-v1.5 — local, completely free |
| **Vector Store** | ChromaDB — local, no API key needed |
| **Python** | 3.11+ |

---

## 🆓 API Keys Needed

| Provider | Cost | Get Key |
|---|---|---|
| **OpenAI** | Paid (you have this) | https://platform.openai.com/api-keys |
| **Groq** | **FREE** | https://console.groq.com |
| **Gemini** | **FREE** tier | https://aistudio.google.com/app/apikey |
| **HuggingFace** | **FREE** | https://huggingface.co/settings/tokens |
| **ChromaDB** | **FREE** local | No key needed |
| **Redis** | **FREE** local | Docker (Phase 5+) |

---

## 🗺️ Build Roadmap — 6 Phases

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6
  Basic       LiteLLM     Automatic   Cost        Caching     Load
  RAG         Gateway     Fallbacks   Tracking    Layer       Balancing
  System      4 providers litellm     completion  litellm     routing
              aliases     .completion _cost()     .Cache()    strategies
                          fallbacks
```

---

## 📐 Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLIENT / API CALLER                               │
│              POST /api/v1/query  ·  POST /api/v1/ingest                 │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │  x-api-key header
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway (app/main.py)                        │
│          Auth · Rate Limiting · Global Exception Handler                 │
│    /api/v1/query  /api/v1/ingest  /api/v1/admin/*  /api/v1/health       │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
┌──────────────────┐              ┌──────────────────────┐
│   RAG Pipeline   │              │   Admin Endpoints     │
│  app/rag/        │              │  Cost · Cache · LB    │
│  pipeline.py     │              │  Health · Budget      │
└────────┬─────────┘              └──────────────────────┘
         │
    ┌────┴──────────────────────────┐
    │                               │
    ▼                               ▼
┌──────────────┐          ┌──────────────────────┐
│  INGESTION   │          │      RETRIEVAL        │
│              │          │                       │
│ loader.py    │          │ embed_query()          │
│ chunker.py   │          │ (local HF model)       │
│ embedder.py  │          │       │                │
│ (local free) │          │       ▼                │
└──────┬───────┘          │  ChromaDB lookup       │
       │                  │  top-k similar chunks  │
       ▼                  └──────────┬─────────────┘
┌──────────────┐                     │
│  ChromaDB    │◄────── store ───────┘
│  (local)     │
│  free, fast  │
└──────────────┘
                                     │  retrieved docs
                                     ▼
                        ┌────────────────────────┐
                        │     GENERATION          │
                        │   generator.py          │
                        └───────────┬────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LiteLLM GATEWAY (app/gateway/)                       │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌───────────┐  │
│  │   router.py  │  │ fallback.py  │  │cost_tracker.py│  │  cache.py │  │
│  │              │  │              │  │               │  │           │  │
│  │ LiteLLM      │  │ litellm      │  │ litellm       │  │ litellm   │  │
│  │ Router()     │  │ .completion  │  │ .completion   │  │ .Cache()  │  │
│  │ routing_     │  │ (fallbacks=) │  │ _cost()       │  │ caching=  │  │
│  │ strategy=    │  │              │  │ token_counter │  │ True      │  │
│  └──────┬───────┘  └──────────────┘  └───────────────┘  └───────────┘  │
│         │                                                                │
│  ┌──────┴──────────────────────────────────────────────────────────┐    │
│  │                  load_balancer.py                                │    │
│  │  simple-shuffle · least-busy · usage-based · latency-based      │    │
│  │  weighted-pick · TPM/RPM limits · circuit breaker               │    │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┬──────────────────┐
           ▼                   ▼                   ▼                  ▼
  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐
  │    OPENAI      │  │     GROQ       │  │   GEMINI     │  │ HUGGINGFACE  │
  │                │  │                │  │              │  │              │
  │ gpt-4o-mini    │  │ llama-3.1-8b   │  │ gemini-1.5   │  │ zephyr-7b    │
  │ gpt-4o         │  │ llama-3.3-70b  │  │ -flash       │  │ phi-3-mini   │
  │                │  │ deepseek-r1    │  │ gemini-2.0   │  │              │
  │ alias: smart   │  │ alias: fast    │  │ -flash-exp   │  │ alias:       │
  │ [paid]         │  │ reasoning      │  │ alias:       │  │ opensource   │
  │                │  │ [FREE]         │  │ balanced     │  │ [FREE]       │
  └────────────────┘  └────────────────┘  │ [FREE]       │  └──────────────┘
                                          └──────────────┘
```

---

## 📁 Production Folder Structure

```
rag-litellm-gateway/
│
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI entry point
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                    # Auth + rate-limit dependencies
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── health.py              # GET  /health
│   │       ├── ingest.py              # POST /ingest/text|url|pdf
│   │       ├── rag.py                 # POST /query  GET /stats
│   │       └── admin.py               # Cost · Cache · LB admin
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                  # All settings (pydantic-settings)
│   │   ├── logger.py                  # Structured JSON logging
│   │   └── exceptions.py             # Full exception hierarchy
│   │
│   ├── gateway/                       # LiteLLM gateway (added Phase 2+)
│   │   ├── __init__.py
│   │   ├── router.py                  # LiteLLM Router — all providers
│   │   ├── fallback.py               # litellm.completion(fallbacks=)
│   │   ├── cost_tracker.py           # litellm.completion_cost()
│   │   ├── budget_manager.py         # litellm.BudgetManager()
│   │   ├── callbacks.py              # litellm CustomLogger callbacks
│   │   ├── cache.py                  # litellm.Cache() setup
│   │   ├── load_balancer.py          # Routing strategies + metrics
│   │   ├── health.py                 # Provider health tracker
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── openai.py             # OpenAI model definitions
│   │       ├── groq.py               # Groq model definitions
│   │       ├── gemini.py             # Gemini model definitions
│   │       └── huggingface.py        # HuggingFace model definitions
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── pipeline.py               # Orchestrates full RAG flow
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py             # PDF · TXT · URL loaders
│   │   │   ├── chunker.py            # RecursiveCharacterTextSplitter
│   │   │   └── embedder.py           # Local HF bge-small embeddings
│   │   ├── vectorstore/
│   │   │   ├── __init__.py
│   │   │   └── chroma.py             # ChromaDB operations
│   │   └── generation/
│   │       ├── __init__.py
│   │       └── generator.py          # LLM answer generation
│   │
│   └── models/
│       ├── __init__.py
│       ├── request.py                # Pydantic request schemas
│       └── response.py              # Pydantic response schemas
│
├── data/
│   ├── raw/                          # Drop source documents here
│   ├── chroma_db/                    # Auto-created by ChromaDB
│   └── samples/
│       └── ai_basics.txt             # Sample document for testing
│
├── tests/
│   ├── conftest.py
│   └── test_phase1.py
│
├── scripts/
│   ├── test_rag.py                   # Phase 1 CLI test
│   ├── test_phase2.py                # Phase 2 multi-provider test
│   ├── test_phase3.py                # Phase 3 fallback test
│   ├── test_phase4.py                # Phase 4 cost tracking test
│   ├── test_phase5.py                # Phase 5 caching test
│   └── test_phase6.py                # Phase 6 load balancing test
│
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── prometheus.yml
│
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🚀 Initial Setup (All Phases)

### 1. Clone & create environment

```bash
git clone https://github.com/yourname/rag-litellm-gateway.git
cd rag-litellm-gateway

python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
# Required for all phases
OPENAI_API_KEY=sk-...

# Required from Phase 2 (all free)
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AI...
HUGGINGFACE_API_KEY=hf_...

GATEWAY_API_KEY=dev-secret-key
```

### 4. Create data directories

```bash
mkdir -p data/{raw,chroma_db,samples}
```

---

---

# Phase 1 — Basic RAG System

## What Phase 1 Builds

A fully working end-to-end RAG pipeline using only:
- **Local HuggingFace embeddings** — free, no API key, runs on CPU
- **ChromaDB** — free local vector database
- **OpenAI gpt-4o-mini** — for answer generation

No LiteLLM yet. Just clean, working RAG you can run and test immediately.

## Phase 1 Architecture

```
 User Question
      │
      ▼
┌─────────────┐    embed_query()     ┌──────────────────────┐
│   FastAPI   │ ──────────────────►  │  Local HF Embedder   │
│  /query     │                      │  BAAI/bge-small      │
└─────────────┘                      │  (free, runs on CPU) │
      │                              └──────────┬───────────┘
      │                                         │ query vector
      │                                         ▼
      │                              ┌──────────────────────┐
      │                              │      ChromaDB        │
      │                              │  cosine similarity   │
      │                              │  top-k retrieval     │
      │                              └──────────┬───────────┘
      │                                         │ top-k chunks
      │                                         ▼
      │                              ┌──────────────────────┐
      │◄─────────── answer ──────────│    OpenAI            │
      │                              │    gpt-4o-mini       │
      ▼                              │    (your key)        │
  Response                           └──────────────────────┘

 Document Ingestion Flow:
 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
 │  loader  │──►│ chunker  │──►│ embedder │──►│ ChromaDB │
 │ PDF/TXT/ │   │ 500 char │   │ local HF │   │ store    │
 │ URL/text │   │ chunks   │   │ vectors  │   │ persist  │
 └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## Phase 1 Files Added

```
app/core/config.py            ← Settings
app/core/logger.py            ← Structured logging
app/core/exceptions.py        ← Exception classes
app/rag/ingestion/loader.py   ← PDF, TXT, URL loaders
app/rag/ingestion/chunker.py  ← Text splitter
app/rag/ingestion/embedder.py ← HuggingFace local embeddings
app/rag/vectorstore/chroma.py ← ChromaDB wrapper
app/rag/generation/generator.py ← OpenAI answer generation
app/rag/pipeline.py           ← Orchestrator
app/api/v1/ingest.py          ← Ingest endpoints
app/api/v1/rag.py             ← Query endpoint
app/main.py                   ← FastAPI app
```

## Run Phase 1

```bash
# Step 1: Add sample document
cat > data/samples/ai_basics.txt << 'EOF'
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
Machine Learning is a subset of AI that enables systems to learn from data.
RAG (Retrieval-Augmented Generation) combines retrieval with language generation.
ChromaDB is a free local vector database for storing embeddings.
The BAAI/bge-small-en-v1.5 model generates embeddings locally for free.
EOF

# Step 2: Test the full pipeline (downloads embedding model ~130MB first time)
python scripts/test_rag.py

# Step 3: Start the API
uvicorn app.main:app --reload --port 8000

# Step 4: Ingest a document
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "RAG combines retrieval with generation to answer questions accurately.",
    "metadata": {"source": "manual", "topic": "AI"}
  }'

# Step 5: Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and how does it work?"}'

# Step 6: Check stats
curl http://localhost:8000/api/v1/stats \
  -H "x-api-key: dev-secret-key"

# Step 7: Run tests
pytest tests/ -v
```

## Phase 1 Sample Response

```json
{
  "question": "What is RAG?",
  "answer": "RAG (Retrieval-Augmented Generation) combines retrieval with language generation to answer questions accurately.",
  "sources": [
    { "text": "RAG combines retrieval...", "metadata": {"source": "manual"}, "score": 0.94 }
  ],
  "model": "gpt-4o-mini",
  "elapsed_s": 1.2
}
```

---

---

# Phase 2 — LiteLLM Gateway + All 4 Providers

## What Phase 2 Adds

Wraps Phase 1's generation layer with **LiteLLM Router**.
Same RAG logic, now routes to any of 4 providers via model aliases.

## Phase 2 Architecture

```
  Query
    │
    ▼
 RAG Pipeline (unchanged)
    │  embed → retrieve → generate
    │
    ▼ model_alias param
┌──────────────────────────────────────────────────────┐
│               LiteLLM Router                          │
│                                                      │
│  "fast"       ──► groq/llama-3.1-8b-instant  [FREE]  │
│  "balanced"   ──► gemini/gemini-1.5-flash    [FREE]  │
│  "reasoning"  ──► groq/deepseek-r1-70b       [FREE]  │
│  "opensource" ──► huggingface/zephyr-7b      [FREE]  │
│  "smart"      ──► gpt-4o-mini               [PAID]  │
│                                                      │
│  routing_strategy = "simple-shuffle"                 │
└──────────────────────────────────────────────────────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
      OpenAI         Groq         Gemini     HuggingFace
```

## Phase 2 Files Added / Modified

```
app/gateway/router.py              ← NEW  LiteLLM Router
app/gateway/providers/openai.py    ← NEW  OpenAI models
app/gateway/providers/groq.py      ← NEW  Groq models (free)
app/gateway/providers/gemini.py    ← NEW  Gemini models (free)
app/gateway/providers/huggingface.py ← NEW HuggingFace models (free)
app/rag/generation/generator.py    ← MOD  use LiteLLM instead of OpenAI SDK
app/rag/pipeline.py                ← MOD  add model_alias param
app/models/request.py              ← MOD  add model_alias to QueryRequest
app/models/response.py             ← MOD  add provider + alias fields
```

## Install Phase 2 Dependencies

```bash
pip install litellm>=1.40.0
```

## Configure Phase 2

Add to your `.env`:

```env
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AI...
HUGGINGFACE_API_KEY=hf_...
DEFAULT_MODEL_ALIAS=fast
```

## Run Phase 2

```bash
# Test all 4 providers with the same query
python scripts/test_phase2.py

# Start API
uvicorn app.main:app --reload --port 8000

# See which providers are loaded
curl http://localhost:8000/api/v1/gateway/info \
  -H "x-api-key: dev-secret-key"

# Query using Groq (free, fastest)
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "model_alias": "fast"}'

# Query using Gemini (free, good quality)
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "model_alias": "balanced"}'

# Query using OpenAI (best writing quality)
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "model_alias": "smart"}'
```

## Phase 2 Model Aliases

| Alias | Model | Provider | Cost | Speed |
|---|---|---|---|---|
| `fast` | llama-3.1-8b-instant | Groq | **FREE** | ⚡ 500+ tok/s |
| `balanced` | gemini-1.5-flash | Gemini | **FREE** | ✅ Good |
| `reasoning` | deepseek-r1-70b | Groq | **FREE** | 🧠 Best reasoning |
| `opensource` | zephyr-7b-beta | HuggingFace | **FREE** | 🐢 Slower |
| `smart` | gpt-4o-mini | OpenAI | Paid | ✍️ Best writing |

---

---

# Phase 3 — Automatic Fallbacks

## What Phase 3 Adds

Uses `litellm.completion()` with the `fallbacks` parameter so if Groq
hits a rate limit, LiteLLM automatically tries Gemini, then OpenAI.
Your code never sees the error.

## Phase 3 Architecture

```
  litellm.completion(
    model  = "groq/llama-3.1-8b",
    messages = [...],
    fallbacks = [              ← Phase 3 key feature
      "gemini/gemini-1.5-flash",
      "gpt-4o-mini"
    ]
  )

  Step 1 ──► groq/llama-3.1-8b
                  ❌ RateLimitError
  Step 2 ──► gemini/gemini-1.5-flash   (1st fallback)
                  ❌ ServiceUnavailable
  Step 3 ──► gpt-4o-mini               (2nd fallback)
                  ✅ Success → return response

  Fallback chains:
  ┌───────────┬──────────────────────────────────────┐
  │  fast     │ → balanced → smart                   │
  │  balanced │ → fast → smart                       │
  │  reasoning│ → balanced → smart                   │
  │  opensource│ → fast → balanced → smart           │
  │  smart    │ → balanced → fast                    │
  └───────────┴──────────────────────────────────────┘

  Router config:
    allowed_fails = 3    → mark unhealthy after 3 fails
    cooldown_time = 60s  → retry after 60 seconds
```

## Phase 3 Files Added / Modified

```
app/gateway/fallback.py      ← NEW  completion_with_fallback()
app/gateway/health.py        ← NEW  provider health tracker
app/gateway/router.py        ← MOD  add fallbacks + context_window_fallbacks
app/rag/generation/generator.py ← MOD  use fallback-aware generation
app/core/exceptions.py       ← MOD  add AllProvidersFailedError
app/models/response.py       ← MOD  add fallback_used + fallback_model fields
app/api/v1/rag.py            ← MOD  add /gateway/health endpoint
```

## Run Phase 3

```bash
# Test fallback scenarios (simulates provider failures)
python scripts/test_phase3.py

# Start API
uvicorn app.main:app --reload --port 8000

# Check provider health + fallback chains
curl http://localhost:8000/api/v1/gateway/health \
  -H "x-api-key: dev-secret-key"

# Query — fallback triggers silently if Groq is down
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "model_alias": "fast"}'
```

## Phase 3 Response (with fallback info)

```json
{
  "question":       "What is RAG?",
  "answer":         "RAG stands for...",
  "model":          "gemini/gemini-1.5-flash",
  "provider":       "gemini",
  "alias":          "fast",
  "fallback_used":  true,
  "fallback_model": "gemini/gemini-1.5-flash",
  "elapsed_s":      1.4
}
```

---

---

# Phase 4 — Cost Tracking with LiteLLM Gateway

## What Phase 4 Adds

Per-request USD cost using `litellm.completion_cost()`,
pre-call token estimation, automatic callbacks, budget manager,
and a full cost summary dashboard endpoint.

## Phase 4 Architecture

```
 Every litellm.completion() call
          │
          ▼
 litellm.completion_cost(response)    ← extracts USD cost
          │
          ▼
 ┌────────────────────────────────────────┐
 │          CostRecord stored             │
 │  model · provider · alias             │
 │  prompt_tokens · completion_tokens    │
 │  total_tokens  · cost_usd             │
 │  fallback_used · question_snippet     │
 └────────────────────────────────────────┘
          │
          ├──► CostStore (in-memory per-session)
          │      total · by_model · by_provider · by_alias
          │
          ├──► BudgetManager (litellm.BudgetManager)
          │      tracks spend · alerts at 80% · blocks at 100%
          │
          └──► Callbacks (litellm CustomLogger)
                 log_success_event fires after every completion

 Pre-call estimation:
   litellm.token_counter(model, messages)  → prompt tokens
   litellm.cost_per_token(model, tokens)   → USD estimate
```

## Phase 4 Files Added / Modified

```
app/gateway/cost_tracker.py      ← NEW  CostStore + extraction functions
app/gateway/budget_manager.py    ← NEW  litellm.BudgetManager wrapper
app/gateway/callbacks.py         ← NEW  litellm CustomLogger
app/gateway/router.py            ← MOD  register_callbacks() at startup
app/gateway/fallback.py          ← MOD  extract cost after completion
app/rag/generation/generator.py  ← MOD  return cost_usd in response
app/models/response.py           ← MOD  add cost_usd field
app/api/v1/admin.py              ← NEW  cost admin endpoints
app/main.py                      ← MOD  register admin router
```

## Run Phase 4

```bash
# Run cost tracking tests
python scripts/test_phase4.py

# Start API
uvicorn app.main:app --reload --port 8000

# Get full cost summary
curl http://localhost:8000/api/v1/admin/cost/summary \
  -H "x-api-key: dev-secret-key"

# Check budget status
curl http://localhost:8000/api/v1/admin/budget \
  -H "x-api-key: dev-secret-key"

# Estimate cost BEFORE calling (saves money)
curl -X POST http://localhost:8000/api/v1/admin/cost/estimate \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain quantum computing in detail", "model_alias": "smart"}'

# Get recent cost records
curl "http://localhost:8000/api/v1/admin/cost/recent?n=5" \
  -H "x-api-key: dev-secret-key"

# Reset cost counters
curl -X POST http://localhost:8000/api/v1/admin/cost/reset \
  -H "x-api-key: dev-secret-key"
```

## Phase 4 LiteLLM Functions Used

| Function | Purpose |
|---|---|
| `litellm.completion_cost(response)` | Get USD cost of completed response |
| `litellm.token_counter(model, messages)` | Count tokens before API call |
| `litellm.cost_per_token(model, tokens)` | Get per-token pricing |
| `litellm.BudgetManager()` | Track and enforce spend limits |
| `litellm.callbacks = [CustomLogger()]` | Auto-capture cost events |

## Phase 4 Cost Summary Response

```json
{
  "total_requests":     10,
  "total_cost_usd":     0.000194,
  "total_tokens":       3240,
  "fallback_requests":  2,
  "avg_cost_per_req":   0.0000194,
  "cost_by_provider":   { "groq": 0.0, "gemini": 0.0, "openai": 0.000194 },
  "cost_by_alias":      { "fast": 0.0, "balanced": 0.0, "smart": 0.000194 }
}
```

---

---

# Phase 5 — Caching with LiteLLM Gateway

## What Phase 5 Adds

Response caching using `litellm.Cache()` with the `caching=True`
parameter on every completion call, plus a custom semantic cache
using local HuggingFace embeddings for similar-question matching.

## Phase 5 Architecture

```
 Query: "What is RAG?"

 Layer 1: Semantic cache (custom, local HF embeddings)
 ─────────────────────────────────────────────────────
 Embed query → cosine similarity search → threshold 0.90
   HIT  → return instantly (0ms, $0.00)
   MISS → continue to Layer 2

 Layer 2: LiteLLM exact cache (litellm.Cache)
 ─────────────────────────────────────────────
 litellm.completion(..., caching=True)
   LiteLLM hashes (model + messages + params)
   HIT  → return cached response instantly
   MISS → call LLM API, store result in cache

 Layer 3: LLM API call
 ──────────────────────
   Call Groq / Gemini / OpenAI
   Store response in both caches

 Cache backends:
 ┌────────────────┬──────────────────┬─────────────────┐
 │ local          │ redis            │ redis-semantic  │
 │ In-memory      │ Persistent       │ Fuzzy matching  │
 │ Dev/testing    │ Production       │ Production      │
 │ No setup       │ needs Redis      │ needs Redis     │
 └────────────────┴──────────────────┴─────────────────┘

 Cache skipped when:
   use_cache=False in request
   alias is in cache_disabled_aliases (e.g. "reasoning")
```

## Phase 5 Files Added / Modified

```
app/gateway/cache.py         ← NEW  litellm.Cache() setup + SemanticCache
app/gateway/router.py        ← MOD  init_litellm_cache() at startup
app/gateway/fallback.py      ← MOD  caching=True in litellm.completion()
app/rag/generation/generator.py ← MOD  return cached flag + use_cache param
app/models/request.py        ← MOD  add use_cache to QueryRequest
app/models/response.py       ← MOD  add cached field + CacheStatsResponse
app/api/v1/admin.py          ← MOD  add cache admin endpoints
app/core/config.py           ← MOD  add cache_type, redis settings
```

## Install Phase 5 Dependencies (Redis — optional)

```bash
# Redis is optional — Phase 5 works with local (in-memory) cache by default
# To use Redis cache:
docker run -d -p 6379:6379 redis:7-alpine
```

## Configure Phase 5

```env
# Default: no Redis needed
CACHE_TYPE=local

# Optional: switch to Redis for persistence
# CACHE_TYPE=redis
# REDIS_HOST=localhost
# REDIS_PORT=6379

CACHE_TTL=3600
SEMANTIC_SIMILARITY_THRESHOLD=0.90
CACHE_DISABLED_ALIASES=["reasoning"]
```

## Run Phase 5

```bash
# Test all caching scenarios
python scripts/test_phase5.py

# Start API
uvicorn app.main:app --reload --port 8000

# Query with cache (default)
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "model_alias": "fast", "use_cache": true}'

# Same question again → should return instantly (cached=true)
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "model_alias": "fast", "use_cache": true}'

# Force fresh LLM call (bypass cache)
curl -X POST http://localhost:8000/api/v1/query \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "model_alias": "fast", "use_cache": false}'

# Cache stats
curl http://localhost:8000/api/v1/admin/cache/stats \
  -H "x-api-key: dev-secret-key"

# Flush all caches
curl -X POST http://localhost:8000/api/v1/admin/cache/flush \
  -H "x-api-key: dev-secret-key"
```

## Phase 5 Cache Response

```json
{
  "question":  "What is RAG?",
  "answer":    "RAG stands for...",
  "model":     "groq/llama-3.1-8b-instant",
  "cached":    true,
  "cost_usd":  0.0,
  "elapsed_s": 0.002
}
```

---

---

# Phase 6 — Load Balancing Across LLM Providers

## What Phase 6 Adds

Full load balancing using all 5 LiteLLM `routing_strategy` options,
per-model TPM/RPM rate limits, circuit breaker pattern,
and a runtime strategy switcher.

## Phase 6 Architecture

```
  Multiple models under same alias "fast":
  ┌────────────────────────────────────────────────────────┐
  │              LiteLLM Router                             │
  │                                                        │
  │  alias: "fast"                                         │
  │    ├── groq/llama-3.1-8b-instant   weight=3  tpm=131k  │
  │    └── groq/llama-3.3-70b-versatile weight=1  tpm=12k  │
  │                                                        │
  │  routing_strategy (pick one):                          │
  │                                                        │
  │  simple-shuffle      → random pick                     │
  │  least-busy          → fewest active requests          │
  │  usage-based-routing → tracks token usage per model    │
  │  latency-based-routing → fastest avg response time     │
  │  weighted-pick       → traffic % by model weight       │
  └────────────────────────────────────────────────────────┘

  Circuit breaker:
  ┌──────────────────────────────────────────────────────┐
  │  Request fails (RateLimitError / Timeout)             │
  │       ├── failure_count += 1                         │
  │       ├── if failure_count >= allowed_fails (3)       │
  │       │       → mark model UNHEALTHY                  │
  │       │       → skip in routing for cooldown_time (60s)│
  │       └── after 60s → mark HEALTHY again              │
  └──────────────────────────────────────────────────────┘

  TPM / RPM limits (per model in provider config):
  ┌─────────────────────────────────────────────────┐
  │  Router tracks usage per model per minute        │
  │  When model hits tpm/rpm limit:                  │
  │    → Router automatically skips to next model    │
  │    → No error seen by caller                     │
  └─────────────────────────────────────────────────┘
```

## Phase 6 Files Added / Modified

```
app/gateway/load_balancer.py  ← NEW  metrics + strategy manager
app/gateway/router.py         ← MOD  routing_strategy + TPM/RPM + circuit breaker
app/gateway/providers/*.py    ← MOD  add tpm + rpm + weight to all models
app/gateway/fallback.py       ← MOD  record per-model latency metrics
app/models/response.py        ← MOD  add routing_strategy field
app/rag/generation/generator.py ← MOD add routing_strategy to response
app/api/v1/admin.py           ← MOD  add LB admin endpoints
app/core/config.py            ← MOD  add routing_strategy + LB settings
```

## Configure Phase 6

```env
# Routing strategy
ROUTING_STRATEGY=usage-based-routing

# Circuit breaker
ALLOWED_FAILS=3
COOLDOWN_TIME=60

# Retry + timeout
NUM_RETRIES=2
REQUEST_TIMEOUT=30
```

## Run Phase 6

```bash
# Run full load balancing test (all 5 strategies + concurrent requests)
python scripts/test_phase6.py

# Start API
uvicorn app.main:app --reload --port 8000

# See current strategy + recommendations
curl http://localhost:8000/api/v1/admin/lb/strategy \
  -H "x-api-key: dev-secret-key"

# View live load metrics
curl http://localhost:8000/api/v1/admin/lb/metrics \
  -H "x-api-key: dev-secret-key"

# Switch strategy at runtime (no restart needed!)
curl -X POST http://localhost:8000/api/v1/admin/lb/strategy/switch \
  -H "x-api-key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"strategy": "latency-based-routing", "reason": "peak hours"}'

# Circuit breaker status
curl http://localhost:8000/api/v1/admin/lb/circuit-breaker \
  -H "x-api-key: dev-secret-key"

# Reset metrics
curl -X POST http://localhost:8000/api/v1/admin/lb/metrics/reset \
  -H "x-api-key: dev-secret-key"
```

## Phase 6 Strategy Guide

| Strategy | Best For | LiteLLM param |
|---|---|---|
| `simple-shuffle` | Dev/testing, low traffic | `routing_strategy="simple-shuffle"` |
| `least-busy` | High concurrency, bursty load | `routing_strategy="least-busy"` |
| `usage-based-routing` | Staying under TPM limits | `routing_strategy="usage-based-routing"` |
| `latency-based-routing` | Latency-sensitive apps | `routing_strategy="latency-based-routing"` |
| `weighted-pick` | A/B testing, gradual rollouts | `routing_strategy="weighted-pick"` |

---

---

## 🔑 Complete API Reference

### Ingest Endpoints

```bash
# Ingest plain text
POST /api/v1/ingest/text
Body: { "text": "...", "metadata": {}, "chunk_size": 500 }

# Ingest from URL
POST /api/v1/ingest/url
Body: { "url": "https://example.com/article" }

# Ingest PDF file
POST /api/v1/ingest/pdf
Body: multipart/form-data  file=@document.pdf
```

### Query Endpoints

```bash
# RAG query
POST /api/v1/query
Body: {
  "question": "What is RAG?",
  "top_k": 4,
  "model_alias": "fast",   # fast|balanced|smart|reasoning|opensource
  "use_cache": true,
  "metadata_filter": {}
}

# Collection stats
GET  /api/v1/stats

# Gateway info (providers, aliases)
GET  /api/v1/gateway/info

# Provider health + fallback chains
GET  /api/v1/gateway/health
```

### Admin Endpoints (Phase 4+)

```bash
# Cost
GET  /api/v1/admin/cost/summary
GET  /api/v1/admin/cost/recent?n=10
POST /api/v1/admin/cost/estimate
POST /api/v1/admin/cost/reset

# Budget
GET  /api/v1/admin/budget

# Latency
GET  /api/v1/admin/latency

# Cache (Phase 5+)
GET  /api/v1/admin/cache/stats
GET  /api/v1/admin/cache/config
POST /api/v1/admin/cache/flush

# Load Balancing (Phase 6+)
GET  /api/v1/admin/lb/metrics
GET  /api/v1/admin/lb/strategy
POST /api/v1/admin/lb/strategy/switch
GET  /api/v1/admin/lb/circuit-breaker
POST /api/v1/admin/lb/metrics/reset
```

---

## 🐳 Docker Deployment

```bash
# Start all services
cd infra
docker compose up --build

# Services:
#   Gateway    → http://localhost:8000
#   Redis      → localhost:6379
#   ChromaDB   → http://localhost:8001
#   Prometheus → http://localhost:9090
```

---

## 🧪 Run All Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing

# Phase-specific scripts
python scripts/test_rag.py       # Phase 1
python scripts/test_phase2.py    # Phase 2
python scripts/test_phase3.py    # Phase 3
python scripts/test_phase4.py    # Phase 4
python scripts/test_phase5.py    # Phase 5
python scripts/test_phase6.py    # Phase 6
```

---

## 📊 LiteLLM Features Used Per Phase

| Phase | LiteLLM Feature | Code |
|---|---|---|
| 2 | Router + model aliases | `Router(model_list=[...])` |
| 2 | Provider abstraction | `litellm_params: {model: "groq/..."}` |
| 3 | Completion fallbacks | `litellm.completion(fallbacks=[...])` |
| 3 | Context window fallback | `context_window_fallbacks=[...]` |
| 3 | Circuit breaker | `allowed_fails=3, cooldown_time=60` |
| 4 | Cost extraction | `litellm.completion_cost(response)` |
| 4 | Token counting | `litellm.token_counter(model, messages)` |
| 4 | Per-token pricing | `litellm.cost_per_token(model, tokens)` |
| 4 | Budget manager | `litellm.BudgetManager()` |
| 4 | Auto callbacks | `litellm.callbacks = [CustomLogger()]` |
| 5 | Cache setup | `litellm.cache = Cache(type="local")` |
| 5 | Cache on completion | `litellm.completion(..., caching=True)` |
| 5 | Redis cache | `Cache(type="redis", host=...)` |
| 5 | Semantic cache | `Cache(type="redis-semantic", similarity_threshold=0.9)` |
| 6 | Routing strategies | `Router(routing_strategy="usage-based-routing")` |
| 6 | TPM/RPM limits | `litellm_params: {tpm: 131072, rpm: 30}` |
| 6 | Model weights | `model_info: {weight: 3}` |

---

## 🛠️ Full Tech Stack

| Layer | Technology | Cost |
|---|---|---|
| LLM Gateway | LiteLLM | Free |
| API Framework | FastAPI + Pydantic v2 | Free |
| Primary LLM | OpenAI gpt-4o-mini | Paid |
| Fast LLM | Groq llama-3.1-8b | **Free** |
| Balanced LLM | Gemini 1.5-flash | **Free** |
| Reasoning LLM | Groq DeepSeek-R1 | **Free** |
| Open-source LLM | HuggingFace Zephyr | **Free** |
| Embeddings | sentence-transformers bge-small | **Free** |
| Vector DB | ChromaDB | **Free** |
| Cache | Redis / in-memory | **Free** |
| Logging | structlog (JSON) | Free |

---

## 📄 License

MIT — free to use, modify, and distribute.
