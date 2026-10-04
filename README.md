# Enterprise RAG Pipeline Handbook

A comprehensive, architecture-first handbook and engineering reference designed to help AI architects, lead engineers, and practitioners master Retrieval-Augmented Generation (RAG) at scale. 

This repository provides production-grade architectural blueprints, clean implementations adhering to SOLID principles, ChromaDB vector indexing, SQLite golden datasets, and automated quantitative evaluation suites across 36 specialized modules.

---

## Target Audience

This handbook is designed for anyone building, evaluating, or scaling production-grade Retrieval-Augmented Generation systems who wants to:

- **Master Architectural Trade-Offs**: Move beyond basic tutorials to understand deep retrieval mechanics, capacity planning, vector memory sizing, and resilient high-availability topologies.
- **Implement Clean, Maintainable Systems**: Build decoupled, testable, and production-ready RAG pipelines adhering to SOLID principles and Clean Architecture.
- **Diagnose & Mitigate Failure Modes**: Systematically resolve hallucinations, context dilution, Lost-in-the-Middle attention bias, and retrieval misses using quantitative benchmark suites.
- **Select the Right Architecture for Specific Workloads**: Make rigorous, data-driven decisions when selecting between Naive RAG, Hybrid Search, Parent-Child Chunking, Graph RAG, Fine-Tuning, or Long-Context models.

---

## Prerequisites

To get the most out of this handbook, readers should have:

- **Python & Modern Software Engineering**: Proficiency in Python 3.11+ (`async`/`await`, typing protocols, dataclasses) and familiarity with Clean Architecture / SOLID design principles.
- **Foundational Machine Learning / NLP**: Basic understanding of embedding vector spaces (cosine similarity, inner products), tokenization (BPE/WordPiece), and Large Language Model inference dynamics (temperature, context windows, prompt formatting).
- **Systems & Database Fundamentals**: Familiarity with relational data modeling (SQL/SQLite), key-value caching (Redis), and core distributed systems concepts (concurrency, throughput/QPS, p95/p99 latency budgets).
- **Development Environment**: Python 3.11+ installed with modern package management (`uv` or `pip`).

---

## Handbook Architecture & Learning Path

| # | Chapter | Core Focus | Architecture & Code |
|---|---|---|---|
| **01** | [Foundations](01-foundations/README.md) | Vector Math, Cosine Similarity & Indexing | [Architecture](01-foundations/architecture/architecture.md) • [Implementation](01-foundations/implementation/retrieval.py) |
| **02** | [End-to-End Architecture](02-end-to-end-architecture/README.md) | Ingestion to Generation Pipeline Decoupling | [Architecture](02-end-to-end-architecture/architecture/architecture.md) • [Implementation](02-end-to-end-architecture/implementation/pipeline.py) |
| **03** | [Document Ingestion](03-document-ingestion/README.md) | Connectors, Deduplication & Hashing | [Architecture](03-document-ingestion/architecture/architecture.md) • [Implementation](03-document-ingestion/implementation/ingestion.py) |
| **04** | [Document Parsing & Multimodal](04-document-parsing-multimodal/README.md) | PDF, Tables, OCR & Layout Extraction | [Architecture](04-document-parsing-multimodal/architecture/architecture.md) • [Implementation](04-document-parsing-multimodal/implementation/parser.py) |
| **05** | [Chunking Fundamentals](05-chunking-fundamentals/README.md) | Fixed, Sentence & Overlap Chunking | [Architecture](05-chunking-fundamentals/architecture/architecture.md) • [Implementation](05-chunking-fundamentals/implementation/chunker.py) |
| **06** | [Advanced Chunking](06-advanced-chunking/README.md) | Semantic, Hierarchical & Parent-Child Chunking | [Architecture](06-advanced-chunking/architecture/architecture.md) • [Implementation](06-advanced-chunking/implementation/semantic_chunker.py) |
| **07** | [Embedding Fundamentals](07-embedding-fundamentals/README.md) | Dimensionality, Distance Metrics & Normalization | [Architecture](07-embedding-fundamentals/architecture/architecture.md) • [Implementation](07-embedding-fundamentals/implementation/embedder.py) |
| **08** | [Embedding Models](08-embedding-models/README.md) | Dense, Sparse & Multi-Vector Embeddings | [Architecture](08-embedding-models/architecture/architecture.md) • [Implementation](08-embedding-models/implementation/embedding_service.py) |
| **09** | [Embedding Selection](09-embedding-selection/README.md) | MTEB Benchmarking, Latency & Cost Trade-offs | [Architecture](09-embedding-selection/architecture/architecture.md) • [Implementation](09-embedding-selection/implementation/selector.py) |
| **10** | [Vector Databases](10-vector-databases/README.md) | HNSW, IVF, Quantization & Index Tuning | [Architecture](10-vector-databases/architecture/architecture.md) • [Implementation](10-vector-databases/implementation/vector_store.py) |
| **11** | [Metadata & Access Control](11-metadata-access-control/README.md) | Multi-Tenancy, RBAC & Hard Filtering | [Architecture](11-metadata-access-control/architecture/architecture.md) • [Implementation](11-metadata-access-control/implementation/access_control.py) |
| **12** | [Retrieval Fundamentals](12-retrieval-fundamentals/README.md) | Top-K, Dense ANN & Similarity Scoring | [Architecture](12-retrieval-fundamentals/architecture/architecture.md) • [Implementation](12-retrieval-fundamentals/implementation/retriever.py) |
| **13** | [Advanced Retrieval](13-advanced-retrieval/README.md) | Hybrid Search (BM25 + Dense) & Reciprocal Rank Fusion | [Architecture](13-advanced-retrieval/architecture/architecture.md) • [Implementation](13-advanced-retrieval/implementation/hybrid_retriever.py) |
| **14** | [Reranking](14-reranking/README.md) | Cross-Encoder Scoring & Two-Stage Cascades | [Architecture](14-reranking/architecture/architecture.md) • [Implementation](14-reranking/implementation/reranker.py) |
| **15** | [Context Engineering](15-context-engineering/README.md) | Lost-in-the-Middle, Compression & Prompt Packing | [Architecture](15-context-engineering/architecture/architecture.md) • [Implementation](15-context-engineering/implementation/context_builder.py) |
| **16** | [Generation & Synthesis](16-generation-synthesis/README.md) | Citation Attribution, Grounded Prompting & Fallbacks | [Architecture](16-generation-synthesis/architecture/architecture.md) • [Implementation](16-generation-synthesis/implementation/synthesizer.py) |
| **17** | [RAG Architectures](17-rag-architectures/README.md) | Modular, Corrective (CRAG), Self-RAG & Adaptive RAG | [Architecture](17-rag-architectures/architecture/architecture.md) • [Implementation](17-rag-architectures/implementation/architectures.py) |
| **18** | [Agentic RAG](18-agentic-rag/README.md) | Multi-Step Reasoning, Routing & Tool Use | [Architecture](18-agentic-rag/architecture/architecture.md) • [Implementation](18-agentic-rag/implementation/agentic_rag.py) |
| **19** | [Evaluation Fundamentals](19-evaluation-fundamentals/README.md) | Ground Truth Creation, Data Drift & Triad Metrics | [Architecture](19-evaluation-fundamentals/architecture/architecture.md) • [Implementation](19-evaluation-fundamentals/implementation/evaluation_pipeline.py) |
| **20** | [Retrieval Evaluation](20-retrieval-evaluation/README.md) | MRR, NDCG@K, Precision@K & Hit Rate Benchmarks | [Architecture](20-retrieval-evaluation/architecture/architecture.md) • [Implementation](20-retrieval-evaluation/implementation/retrieval_evaluator.py) |
| **21** | [Generation Evaluation](21-generation-evaluation/README.md) | Faithfulness, Groundedness & Hallucination Scoring | [Architecture](21-generation-evaluation/architecture/architecture.md) • [Implementation](21-generation-evaluation/implementation/generation_evaluator.py) |
| **22** | [LLM-as-a-Judge](22-llm-as-judge/README.md) | Position Bias, G-Eval & Pairwise Verification | [Architecture](22-llm-as-judge/architecture/architecture.md) • [Implementation](22-llm-as-judge/implementation/judge.py) |
| **23** | [Evaluation Frameworks](23-evaluation-frameworks/README.md) | RAGAS & TruLens Metric Integration | [Architecture](23-evaluation-frameworks/architecture/architecture.md) • [Implementation](23-evaluation-frameworks/implementation/framework_adapter.py) |
| **24** | [Evaluation Datasets](24-evaluation-datasets/README.md) | Synthetic QA Generation & SQLite Golden Corpus | [Architecture](24-evaluation-datasets/architecture/architecture.md) • [Implementation](24-evaluation-datasets/implementation/dataset_generator.py) |
| **25** | [RAG Failure Modes](25-rag-failure-modes/README.md) | 7 Critical Failure Modes & Guardrail Recovery | [Architecture](25-rag-failure-modes/architecture/architecture.md) • [Implementation](25-rag-failure-modes/implementation/circuit_breaker.py) |
| **26** | [Production RAG](26-production-rag/README.md) | High-Concurrency Batching, Fallbacks & Circuit Breakers | [Architecture](26-production-rag/architecture/architecture.md) • [Implementation](26-production-rag/implementation/production_service.py) |
| **27** | [RAG Observability](27-rag-observability/README.md) | OpenTelemetry Tracing, Latency Attribution & Logging | [Architecture](27-rag-observability/architecture/architecture.md) • [Implementation](27-rag-observability/implementation/tracer.py) |
| **28** | [Performance Optimization](28-performance-optimization/README.md) | TTFT Optimization, Semantic Caching & Speculative RAG | [Architecture](28-performance-optimization/architecture/architecture.md) • [Implementation](28-performance-optimization/implementation/cache_optimizer.py) |
| **29** | [RAG Security](29-rag-security/README.md) | Indirect Prompt Injection, Exfiltration & PII Scrubbing | [Architecture](29-rag-security/architecture/architecture.md) • [Implementation](29-rag-security/implementation/security_guardrail.py) |
| **30** | [Graph RAG](30-graph-rag/README.md) | Knowledge Graph Extraction & Hybrid Community Queries | [Architecture](30-graph-rag/architecture/architecture.md) • [Implementation](30-graph-rag/implementation/graph_rag.py) |
| **31** | [LlamaIndex Deep Dive](31-llamaindex-deep-dive/README.md) | Query Engines, Custom Node Parsers & Workflows | [Architecture](31-llamaindex-deep-dive/architecture/architecture.md) • [Implementation](31-llamaindex-deep-dive/implementation/llamaindex_pipeline.py) |
| **32** | [Enterprise RAG Project](32-enterprise-rag-project/README.md) | Multi-Source SEC 10-K Ingestion & Aggregation | [Architecture](32-enterprise-rag-project/architecture/architecture.md) • [Implementation](32-enterprise-rag-project/implementation/enterprise_orchestrator.py) |
| **33** | [RAG Case Studies](33-rag-case-studies/README.md) | Healthcare, Legal, Customer Support & Financial Blueprints | [Architecture](33-rag-case-studies/architecture/architecture.md) • [Implementation](33-rag-case-studies/implementation/case_study_engine.py) |
| **34** | [RAG System Design & Sizing](34-rag-system-design-architecture/README.md) | QPS, Sharding, RAM/VRAM Sizing & High Availability | [Architecture](34-rag-system-design-architecture/architecture/architecture.md) • [Implementation](34-rag-system-design-architecture/implementation/sizing_calculator.py) |
| **35** | [RAG Decision Framework](35-rag-decision-framework/README.md) | Decision Matrix: RAG vs Fine-Tuning vs Long Context | [Architecture](35-rag-decision-framework/architecture/architecture.md) • [Implementation](35-rag-decision-framework/implementation/decision_matrix.py) |
| **36** | [Production Checklist](36-production-checklist/README.md) | 50-Point Pre-Flight Production Readiness Gate | [Architecture](36-production-checklist/architecture/architecture.md) • [Implementation](36-production-checklist/implementation/readiness_auditor.py) |

---

## Architectural Taxonomy & Mental Model

```
       ┌─────────────────────────────────────────────────────────┐
       │                   Ingestion Pipeline                    │
       │  Docs ──► Extract ──► Clean ──► Chunk ──► Embed ──► DB  │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                    Online RAG Engine                    │
       │                                                         │
       │   Query ──► Guardrail ──► Query Transformation          │
       │                                  │                      │
       │                                  ▼                      │
       │                    ┌─────────────────────────┐          │
       │                    │    Hybrid Retrieval     │          │
       │                    │  (Dense ANN + BM25 RRF) │          │
       │                    └─────────────┬───────────┘          │
       │                                  │                      │
       │                                  ▼                      │
       │                    ┌─────────────────────────┐          │
       │                    │  Cross-Encoder Rerank   │          │
       │                    └─────────────┬───────────┘          │
       │                                  │                      │
       │                                  ▼                      │
       │                    ┌─────────────────────────┐          │
       │                    │   Context Compression   │          │
       │                    └─────────────┬───────────┘          │
       │                                  │                      │
       │                                  ▼                      │
       │                    ┌─────────────────────────┐          │
       │                    │ Attributed Generation   │          │
       │                    └─────────────┬───────────┘          │
       └──────────────────────────────────┼──────────────────────┘
                                          │
                                          ▼
       ┌─────────────────────────────────────────────────────────┐
       │              Continuous Evaluation & Telemetry          │
       │  Faithfulness • Relevance • Latency • Cost • Drifts     │
       └─────────────────────────────────────────────────────────┘
```

---

## Module Anatomy

Every module follows a decoupled Clean Architecture standard:

```
<module-directory>/
├── pyproject.toml              # Independent dependency declaration
├── README.md                   # Chapter entry point (Architecture, Tradeoffs, Failure Modes, Decision Matrix)
├── architecture/
│   └── architecture.md         # Mental model diagram & SOLID design rationale
├── implementation/
│   └── <service>.py            # SOLID implementation using ChromaDB & Abstract Base Classes
├── data/
│   └── golden_dataset.py       # SQLite database schema, seeding, and ground-truth repository
└── evaluation/
    └── evaluator.py            # Automated evaluation runner against SQLite golden dataset
```

---

## Quick Start & Running Evaluations

Each module is self-contained. You can navigate into any chapter and execute its automated evaluation suite against its SQLite golden dataset:

```bash
# Example: Run the Advanced Hybrid Retrieval benchmark (Chapter 13)
cd 13-advanced-retrieval
python evaluation/evaluator.py

# Example: Run unit tests and SOLID protocol verifications
pytest evaluation/
```

---

## Core Engineering Principles

1. **Information Retrieval First**: Never attempt to tune prompt instructions or switch generation models before verifying that the retrieval recall and reranker top-K precision are mathematically sufficient.
2. **Strict Protocol Decoupling**: All components communicate through interfaces/protocols (`typing.Protocol` / `abc.ABC`), ensuring modularity, easy swapping of models, and test isolation.
3. **Reproducible Golden Evaluation**: Every architectural decision is benchmarked against versioned ground-truth datasets stored in SQLite.
4. **Resilience & Defensiveness**: Production systems must incorporate timeouts, circuit breakers, fallback degradation strategies, and input/output sanitization.
