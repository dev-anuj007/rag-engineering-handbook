# Foundations Architecture & Mental Model

## End-to-End Foundations Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Offline Ingestion Pipeline               │
                    │                                                        │
                    │  Raw Documents ──► Text Normalizer ──► Chunking Engine │
                    │                                              │         │
                    │                                              ▼         │
                    │  ChromaDB Index ◄── Normalized Embeddings ◄── Embedder │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │                Online Query Engine                     │
                    │                                                        │
                    │  User Query ──► Query Normalizer ──► Dense Embedder    │
                    │                                             │          │
                    │                                             ▼          │
                    │  Top-K Chunks ◄── Cosine Similarity ANN ◄── ChromaDB   │
                    │        │                                               │
                    │        ▼                                               │
                    │  Context Assembler ──► Grounded LLM Prompt ──► Answer  │
                    └────────────────────────────────────────────────────────┘
```

---

## SOLID Principles in RAG Foundations

1. **Single Responsibility Principle (SRP)**:
   - `EmbeddingServiceProtocol`: Dedicated solely to converting string payloads into normalized floating-point dense vectors.
   - `VectorRepositoryProtocol`: Responsible strictly for indexing and executing $K$-NN / Cosine similarity vector queries.
   - `GroundedSynthesizer`: Handles context-to-prompt assembly and bounded LLM synthesis without coupling to retrieval storage.

2. **Open/Closed Principle (OCP)**:
   - Extending similarity metrics or replacing local in-memory embeddings with cloud API embeddings (e.g. Gemini / OpenAI) requires implementing `EmbeddingServiceProtocol` without altering query parsing or database insertion logic.

3. **Liskov Substitution Principle (LSP)**:
   - Vector repositories (ChromaDB, SQLite vector extensions, in-memory numpy) adhere strictly to `VectorRepositoryProtocol` and can be substituted without affecting higher-level orchestrators.

4. **Interface Segregation Principle (ISP)**:
   - Separate fine-grained protocols for `VectorReader` (search-only) and `VectorWriter` (ingestion/mutation), preventing query-time microservices from requiring write access.

5. **Dependency Inversion Principle (DIP)**:
   - High-level pipeline orchestrators depend strictly on abstract protocols (`EmbeddingServiceProtocol`, `VectorRepositoryProtocol`), never on concrete third-party SDK clients.
