# BPMN-RAG

> **AI-powered BPMN Process Analysis & Improvement Assistant**
> Upload BPMN models, ask questions, compare processes, and get concrete improvement recommendations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io/)
[![Neo4j](https://img.shields.io/badge/Neo4j-6.0+-008CC1?logo=neo4j)](https://neo4j.com/)
[![bpmn2neo](https://img.shields.io/badge/bpmn2neo-0.1.9-green.svg)](https://github.com/alciakng/bpmn2neo)

---

## What is BPMN-RAG?

**BPMN-RAG** is an intelligent Graph-RAG agent that transforms BPMN process diagrams into actionable insights.

### Key Capabilities

- **Upload & Index**: Ingest `.bpmn` files into Neo4j with automatic vector embeddings
- **Semantic Search**: Find relevant processes using hybrid retrieval (cosine similarity + context aggregation)
- **Natural Language Q&A**: Ask questions in Korean or English about your processes
- **Process Comparison**: Compare uploaded model vs existing models for gap analysis
- **Improvement Recommendations**: Get concrete, KPI-driven suggestions for process optimization
- **Interactive Graph Viz**: Visualize BPMN elements with hierarchical relationships

---

## Architecture

```
┌─────────────┐
│ Streamlit   │  User uploads BPMN + asks questions
│     UI      │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│           GraphQueryAgent                    │
│  ┌──────────────┐  ┌─────────────────────┐   │
│  │   Ingest     │  │  Q&A Pipeline       │   │
│  │              │  │                     │   │
│  │ bpmn2neo     │  │ QueryInterpreter    │   │
│  │ • Parse      │  │ • Embed query       │   │
│  │ • Load       │  │ • Hybrid search     │   │
│  │ • Embed      │  │ • Re-rank           │   │
│  │              │  │                     │   │
│  │              │  │ ContextComposer     │   │
│  │              │  │ • Fetch context     │   │
│  │              │  │ • Build payload     │   │
│  │              │  │                     │   │
│  │              │  │ LLM (GPT-4o)        │   │
│  │              │  │ • Generate answer   │   │
│  │              │  │ • Recommend fixes   │   │
│  └──────────────┘  └─────────────────────┘   │
└──────────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Neo4j Graph   │
              │                 │
              │ • BPMN nodes    │
              │ • Embeddings    │
              │ • Relationships │
              └─────────────────┘
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **GraphQueryAgent** | Orchestrates end-to-end workflow: ingestion → retrieval → answering |
| **QueryInterpreter** | Embeds queries, performs hybrid search, re-ranks by cosine similarity |
| **ContextComposer** | Fetches hierarchical BPMN context (Model → Participant → Process → Lane → FlowNode) |
| **Reader** | Neo4j access layer with Cypher queries for retrieval and context fetching |
| **UI Components** | Streamlit-based chat interface, BPMN uploader, and interactive graph visualization |

---

## Installation

### Prerequisites

- **Python** 3.10+
- **Neo4j** 6.0+ (with GDS library for cosine similarity)
- **OpenAI API Key** (for embeddings and GPT-4o)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BPMN-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   NEO4J_URI = "bolt://localhost:7687"
   NEO4J_USERNAME = "neo4j"
   NEO4J_PASSWORD = "your_password"
   NEO4J_DATABASE = "neo4j"

   OPENAI_API_KEY = "sk-..."

   # Optional: AWS S3 for image storage
   AWS_ACCESS_KEY_ID = "..."
   AWS_SECRET_ACCESS_KEY = "..."
   AWS_DEFAULT_REGION = "ap-northeast-2"
   S3_BUCKET_NAME = "your-bucket"
   ```

4. **Launch the app**
   ```bash
   streamlit run main.py
   ```

---

## Usage

### 1. Upload BPMN Models

- Navigate to **BPMN Loader** page
- Upload `.bpmn` file (optionally with process image)
- Click **적재하기** (Ingest)
- Model is parsed, loaded into Neo4j, and embedded using `bpmn2neo`

### 2. Ask Questions

- Go to **Chat** page
- Type natural language queries:
  - Where are the bottlenecks?
  - How many lane handoffs occur?
  - How do data objects flow?

### 3. Compare Processes

- Upload a target BPMN model
- Ask comparison questions:
  - What are the differences?
  - What improvements can be made?

### 4. Get Recommendations

The agent generates:
- **Problem Diagnosis Table**: Issues, evidence (node IDs), severity, solutions
- **Improvements & Effects Table**: Actions, KPIs, baseline → target, expected delta (%)
- **Risk & Recommendations**: Actionable mitigations with process-specific context

---

## Features in Detail

### Hybrid Retrieval Pipeline

1. **Query Embedding**: Encode user query with OpenAI `text-embedding-3-small`
2. **Cosine Search**: Retrieve top-N FlowNodes using `gds.similarity.cosine`
3. **Context Aggregation**: Group by model, aggregate scores from child nodes
4. **Re-ranking**: Select top-K models with highest cumulative scores

### LLM Prompt Engineering

- **System Prompt**: Configures GPT-4o as "BPMN/Neo4j Graph-RAG expert + Process Innovation Consultant"
- **Payload Schema**: Structured JSON with model/participant/process/lane/node hierarchy
- **Chat History**: Maintains last 3 turns for continuity
- **Style Rules**:
  - Korean output with numbered sections, bullet points, tables
  - Inline code for domain terms (e.g., `Bank Branch (Front Office)`, `Underwriter`)
  - Pure Markdown (no HTML)

### Graph Visualization

- **Interactive Graph**: Streamlit-Agraph for node/edge rendering
- **Multi-Model Support**: Slider to select/compare multiple loaded models
- **Hierarchical Layout**: Process → Lane → FlowNode structure preserved

---

## Project Structure

```
BPMN-RAG/
├── agent/
│   ├── graph_query_agent.py    # Main orchestrator (ingest, derive, answer)
│   ├── query_interpreter.py    # Query embedding + hybrid search + re-ranking
│   └── context_composer.py     # Fetch hierarchical context from Neo4j
├── manager/
│   ├── reader.py                # Neo4j Cypher queries (search, fetch)
│   ├── session_store.py         # Session state management
│   └── util.py                  # Utility functions
├── ui/
│   ├── app/
│   │   ├── init.py              # App initialization (agent, session)
│   │   └── handler.py           # Business logic handlers
│   ├── component/
│   │   ├── chat.py              # Chat interface
│   │   ├── uploader.py          # BPMN/image uploader
│   │   ├── agraph.py            # Graph visualization
│   │   ├── panels.py            # Candidate selector panel
│   │   └── layout.py            # App layout structure
│   └── common/
│       ├── utils.py             # UI utilities
│       └── log_viewer.py        # Live log streaming
├── common/
│   ├── settings.py              # Configuration (Neo4j, OpenAI)
│   ├── neo4j_repo.py            # Neo4j client wrapper
│   ├── llm_client.py            # LLM client (OpenAI)
│   └── logger.py                # Structured logging
├── .streamlit/
│   ├── config.toml              # Streamlit config
│   └── secrets.toml             # Secrets (not in repo)
├── main.py                      # Streamlit entry point
└── requirements.txt             # Python dependencies
```

---

## Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| `bpmn2neo` | 0.1.9 | BPMN parsing, Neo4j loading, embedding |
| `streamlit` | 1.50.0 | Web UI framework |
| `neo4j` | 6.0.2 | Graph database driver |
| `openai` | 2.6.1 | LLM + embedding API |
| `streamlit-agraph` | 0.0.45 | Interactive graph visualization |
| `redis` | 7.0.0 | Session caching (optional) |
| `boto3` | 1.40.23 | AWS S3 for image storage (optional) |

---

## Neo4j Setup

### Required Indexes

```cypher
// Fulltext index for keyword search
CREATE FULLTEXT INDEX nodeTextIndex IF NOT EXISTS
FOR (n:Activity|Event|Gateway)
ON EACH [n.name, n.summary_text, n.full_text];

// Node key constraints (created by bpmn2neo)
CREATE CONSTRAINT IF NOT EXISTS FOR (n:BPMNModel) REQUIRE (n.id, n.modelKey) IS NODE KEY;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Process) REQUIRE (n.id, n.modelKey) IS NODE KEY;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Lane) REQUIRE (n.id, n.modelKey) IS NODE KEY;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Activity) REQUIRE (n.id, n.modelKey) IS NODE KEY;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Event) REQUIRE (n.id, n.modelKey) IS NODE KEY;
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Gateway) REQUIRE (n.id, n.modelKey) IS NODE KEY;
```

### Required GDS Library

Ensure Neo4j Graph Data Science library is installed for `gds.similarity.cosine()`:
- [Neo4j GDS Installation Guide](https://neo4j.com/docs/graph-data-science/current/installation/)

---

## Example Queries

### Process Analysis
- What are the main steps?
- Identify bottleneck points
- How are data objects used?

### Comparison & Improvement
- Suggest improvements for uploaded model
- What are differences vs existing models?
- How to reduce lead time?

### Risk & Compliance
- What are compliance risks?
- Are there missing approval steps?

---

## Configuration

### Embedding Settings

Adjust in `common/settings.py`:
```python
@dataclass
class OpenAIConfig:
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536  # or 3072 for -large
    translation_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens_full: int = 600
```

### Retrieval Tuning

Adjust in `agent/query_interpreter.py`:
```python
top_k_nodes_per_model: int = 10   # Top-K nodes per model
top_n_models: int = 100           # Top-N models to return
wc: float = 0.45                  # Cosine weight
wb: float = 0.40                  # BM25 weight
wd: float = 0.15                  # Data I/O weight
```

---

## Troubleshooting

### Neo4j Connection Errors
- Verify `NEO4J_URI` in `.streamlit/secrets.toml`
- Check firewall and Neo4j authentication
- Ensure database is running and accessible

### OpenAI API Errors
- Validate `OPENAI_API_KEY` is correct
- Check API rate limits and quotas
- Monitor token usage for large payloads

### Empty Search Results
- Ensure BPMN models have been embedded (`mode='light'` or `mode='all'`)
- Check `context_vector` property exists on FlowNodes
- Verify GDS library is installed for cosine similarity

### Graph Visualization Issues
- Reduce number of nodes if graph is too large
- Check Neo4j query returns valid node/relationship data
- Clear browser cache if layout is broken

---

## Performance

### Typical Processing Times

| Operation | Time | Notes |
|-----------|------|-------|
| BPMN Ingestion | 10-30s | Depends on model size |
| Embedding (light mode) | 5-15s | FlowNodes only |
| Embedding (full mode) | 30-60s | All hierarchy levels |
| Query + Retrieval | 1-3s | Top-200 candidates |
| LLM Answer Generation | 5-15s | GPT-4o completion |

### Cost Estimates

- **Embedding**: ~$0.0001 per 1K tokens (text-embedding-3-small)
- **LLM**: ~$0.015 per 1K tokens (gpt-4o input/output combined)
- **Typical Query**: ~$0.05-0.20 per complete Q&A cycle

---

## Limitations

- **Language**: Primarily optimized for Korean output (English supported but less tuned)
- **Model Size**: Very large BPMN diagrams (>500 nodes) may hit token limits
- **Quantification**: KPI improvements are estimated; actual impact requires validation
- **Real-time**: Not designed for streaming/incremental updates during process execution

---

## Roadmap

- [ ] Multi-language support (English, Japanese)
- [ ] BPMN 2.0 validation and linting
- [ ] Process simulation integration (token replay)
- [ ] Custom KPI calculators (cycle time, wait time, handoffs)
- [ ] Export recommendations as BPMN annotations
- [ ] Batch comparison mode (N vs M models)

---

## License

This project builds on [bpmn2neo](https://github.com/alciakng/bpmn2neo) (Apache 2.0).
Refer to individual dependency licenses for compliance.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with clear description

---

## Acknowledgments

- **bpmn2neo**: BPMN parsing and Neo4j loading library
- **Streamlit**: Rapid UI prototyping framework
- **Neo4j**: Graph database and GDS library
- **OpenAI**: Embedding and LLM APIs

---

**Built with for Process Innovation**
