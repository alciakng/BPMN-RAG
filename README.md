# BPMN-RAG

> **AI-powered BPMN Process Analysis & Improvement Assistant**
> Upload BPMN models, ask questions, compare processes, and get concrete improvement recommendations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io/)
[![Neo4j](https://img.shields.io/badge/Neo4j-6.0+-008CC1?logo=neo4j)](https://neo4j.com/)
[![bpmn2neo](https://img.shields.io/badge/bpmn2neo-0.2.4-green.svg)](https://github.com/alciakng/bpmn2neo)

---

## What is BPMN-RAG?


### Youtube Link
[![Vidio Label](http://img.youtube.com/vi/N5x9oLt1wZE/0.jpg)](https://www.youtube.com/watch?v=N5x9oLt1wZE)

**BPMN-RAG** is an intelligent Graph-RAG agent that transforms BPMN process diagrams into actionable insights.

### Key Capabilities

- **Upload & Index**: Ingest `.bpmn` files into Neo4j with automatic vector embeddings
- **Semantic Search**: Find relevant processes using hybrid retrieval (cosine similarity + minimum threshold filtering)
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
┌─────────────────────────────────────────────────────────────┐
│                    GraphQueryAgent                          │
│                                                             │
│  ┌──────────────┐                ┌─────────────────────┐    │
│  │   Ingest     │                │   Q&A Pipeline      │    │
│  │              │                │                     │    │
│  │  bpmn2neo    │                │  QueryInterpreter   │    │
│  │  • Parse     │                │  • Embed query      │    │
│  │  • Load      │                │  • Search           │    │
│  │  • Embed     │                │  • Re-rank          │    │
│  │              │                │                     │    │
│  └──────┬───────┘                │  ContextComposer    │    │
│         │                        │  • Fetch context    │    │
│         │                        │  • Model flows      │    │
│         │                        │  • Build payload    │    │
│         │                        └──────────┬──────────┘    │
│         │                                   │               │
│         ▼                                   ▼               │
│  ┌──────────────┐                  ┌─────────────────┐      │
│  │  Neo4j       │◄─────Retrieval───│  LLM (GPT-4.1)  │      │
│  │  Graph       │                  │  • Analyze      │      │
│  │              │                  │  • Answer       │      │
│  │ • BPMN nodes │                  │  • Recommend    │      │
│  │ • Embeddings │                  └────────┬────────┘      │
│  │ • Flows      │                           │               │
│  └──────────────┘                           │               │
└─────────────────────────────────────────────┼───────────────┘
                                              ▼
                                       ┌─────────────┐
                                       │   Answer    │
                                       └─────────────┘
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **GraphQueryAgent** | Orchestrates end-to-end workflow: ingestion → retrieval → answering |
| **QueryInterpreter** | Embeds queries, performs hybrid search with cosine similarity, re-ranks by score |
| **ContextComposer** | Fetches hierarchical BPMN context (Model → Participant → Process → Lane → FlowNode) and 4-hop model flows |
| **Reader** | Neo4j access layer with Cypher queries for retrieval and context fetching with minimum similarity threshold |
| **UI Components** | Streamlit-based chat interface, BPMN uploader, and interactive graph visualization |

---

## Installation

### Prerequisites

- **Python** 3.10+
- **Neo4j** 6.0+ (with GDS library for cosine similarity)
- **OpenAI API Key** (for embeddings and GPT-4.1)

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

   # Optional: Redis for session management
   REDIS_URL = "redis://localhost:6379"

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

- Navigate to **BPMN 적재** (BPMN Loader) page
- Upload `.bpmn` file (optionally with process image)
- Click **적재하기** (Ingest)
- Model is parsed, loaded into Neo4j, and embedded using `bpmn2neo`

### 2. Ask Questions

- Go to **프로세스 분석** (Process Analysis) page
- Type natural language queries:
  - Can you explain the social insurance premium calculation process in the payroll payment process?
  - What are the improvements for this process?
  - Can you write the improvements for this process as BPMN XML code?

### 3. Compare Processes

- Upload a target BPMN model
- Ask comparison questions:
  - What are the differences?
  - What improvements can be made?

### 4. Get Recommendations

The agent generates:
- **Process Overview**: General explanation of the process with context
- **Process Flow Diagram**: Visual representation with predecessor/successor models
- **Problem Diagnosis Table**: Issues, evidence (node IDs), severity, solutions
- **Improvements & Effects Table**: Actions, KPIs, baseline → target, expected delta (%)
- **Risk & Recommendations**: Actionable mitigations with process-specific context

---

## Features in Detail

### Hybrid Retrieval Pipeline

1. **Query Embedding**: Encode user query with OpenAI `text-embedding-3-large`
2. **Cosine Search**: Retrieve top-N FlowNodes using `gds.similarity.cosine`
3. **Minimum Similarity Filtering**: Filter results by minimum cosine similarity threshold (default: 0.3)
4. **Context Aggregation**: Group by model, aggregate scores from child nodes
5. **Re-ranking**: Select top-K models with highest cumulative scores

### LLM Prompt Engineering

- **System Prompt**: Configures GPT-4.1 as "BPMN/Neo4j Graph-RAG expert + Process Innovation Consultant"
- **Payload Schema**: Structured JSON with model/participant/process/lane/node hierarchy and model_flows
- **Model Flows**: 4-hop predecessor/successor process chains for context understanding
- **Chat History**: Maintains last 3 turns for continuity
- **Style Rules**:
  - Korean output with numbered sections, bullet points, tables
  - Inline code for domain terms (e.g., `Bank Branch (Front Office)`, `Underwriter`)
  - Pure Markdown (no HTML)
  - Process Flow Diagrams with arrows and role explanations

### Graph Visualization

- **Interactive Graph**: Streamlit-Agraph for node/edge rendering
- **Multi-Model Support**: Slider to select/compare multiple loaded models
- **Hierarchical Layout**: Category → Model → Process → Lane → FlowNode structure preserved
- **Category Tree Viewer**: Hierarchical category and model navigation

---

## Project Structure

```
BPMN-RAG/
├── agent/
│   ├── graph_query_agent.py      # Main orchestrator (ingest, derive, answer)
│   ├── query_interpreter.py      # Query embedding + hybrid search + re-ranking
│   └── context_composer.py       # Fetch hierarchical context from Neo4j
├── manager/
│   ├── reader.py                  # Neo4j Cypher queries (search, fetch, model flows)
│   ├── session_store.py           # Session/analysis state management (Redis/memory)
│   ├── vector_store.py            # FAISS vector store for embeddings
│   └── util.py                    # Utility functions
├── ui/
│   ├── app/
│   │   ├── init.py                # App initialization (agent, session, stores)
│   │   └── handler.py             # Business logic handlers
│   ├── component/
│   │   ├── chat.py                # Chat interface with history
│   │   ├── chat_input_module.py   # Chat input component
│   │   ├── uploader.py            # BPMN/image uploader with category selection
│   │   ├── agraph.py              # Interactive graph visualization
│   │   ├── category_viewer.py     # Category tree navigation
│   │   ├── panels.py              # Model selection sidebar panel
│   │   ├── intro.py               # Introduction page
│   │   ├── layout.py              # App layout structure with sidebar menu
│   │   └── common/
│   │       └── tree_viewer.py     # Hierarchical tree component
│   └── common/
│       ├── utils.py               # UI utilities
│       ├── analytics.py           # Analytics tracking
│       └── log_viewer.py          # Live log streaming
├── common/
│   ├── settings.py                # Configuration (Neo4j, OpenAI)
│   ├── neo4j_repo.py              # Neo4j client wrapper
│   ├── llm_client.py              # LLM client (OpenAI)
│   ├── logger.py                  # Structured logging
│   └── util.py                    # Common utilities
├── .streamlit/
│   ├── config.toml                # Streamlit config
│   └── secrets.toml               # Secrets (not in repo)
├── main.py                        # Streamlit entry point
└── requirements.txt               # Python dependencies
```

---

## Requirements

| Dependency | Version | Purpose |
|------------|---------|---------|
| `bpmn2neo` | 0.2.4 | BPMN parsing, Neo4j loading, embedding with NEXT_PROCESS support |
| `streamlit` | 1.50.0 | Web UI framework |
| `streamlit-option-menu` | 0.4.0 | Sidebar navigation menu |
| `streamlit-agraph` | 0.0.45 | Interactive graph visualization |
| `streamlit-extras` | 0.5.1 | Additional UI components |
| `neo4j` | 6.0.2 | Graph database driver |
| `openai` | 2.6.1 | LLM + embedding API (GPT-4.1, text-embedding-3-large) |
| `redis` | 7.0.0 | Session caching and state management |
| `faiss-cpu` | 1.9.0.post1 | Vector similarity search |
| `boto3` | 1.40.23 | AWS S3 for image storage (optional) |
| `pytz` | 2024.2 | Timezone utilities |

---

## Configuration

### Embedding Settings

Adjust in [common/settings.py](common/settings.py):
```python
@dataclass
class OpenAIConfig:
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    translation_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens_full: int = 600
```

### Retrieval Tuning

Adjust in [agent/query_interpreter.py](agent/query_interpreter.py):
```python
top_k_nodes_per_model: int = 10   # Top-K nodes per model
top_n_models: int = 100           # Top-N models to return
wc: float = 0.45                  # Cosine weight
wb: float = 0.40                  # BM25 weight (if using hybrid search)
wd: float = 0.15                  # Data I/O weight (if using hybrid search)
```

### Minimum Similarity Threshold

Adjust in [manager/reader.py](manager/reader.py):
```python
def search_candidates(
    self,
    user_query: str,
    qemb: Optional[List[float]],
    limit: int = 200,
    min_similarity: float = 0.3  # Adjust threshold (0.0 ~ 1.0)
):
```

Higher values (e.g., 0.5) return only highly relevant results, while lower values (e.g., 0.2) allow more diverse candidates.

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
- Lower `min_similarity` threshold if results are too restrictive

### Graph Visualization Issues
- Reduce number of nodes if graph is too large
- Check Neo4j query returns valid node/relationship data
- Clear browser cache if layout is broken

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
