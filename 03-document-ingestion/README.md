# 03 Document Ingestion

## Objective

Build a robust, deduplicated document ingestion subsystem that normalizes metadata, prevents duplicate embeddings in ChromaDB, and verifies ingestion accuracy against SQLite benchmarks.

## Why It Matters

Ingesting redundant or corrupt documents pollutes the vector index, skews similarity calculations, inflates hosting costs, and introduces duplicate context during LLM generation.

## Core Concepts

- **Cryptographic Content Hashing**: SHA-256 fingerprinting ensures idempotency across ingestion batches.
- **Deduplication Store**: Decoupled registry tracking ingested documents before vector computation.
- **ChromaDB Vector Store**: Structured vector insertion with rich metadata enrichment.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Ingestion Topology

Enterprise document ingestion is an asynchronous ETL (Extract, Transform, Load) pipeline that transforms raw unstructured files into sanitized, deduplicated, and vector-indexed entities.

```
                    ┌────────────────────────────────────────┐
                    │ Raw Document Source (S3, GCS, Confluence)
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Content Hashing & Deduplication Layer  │
                    │   Compute SHA-256(RawBytes + Metadata) │
                    └───────────────────┬────────────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     │ Hash Exists in DB?                  │
                     ├──────────────────┬──────────────────┤
                     │ YES              │ NO               │
                     │ (Skip / No-Op)   │ (Process File)   │
                     └──────────────────┼──────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ File Format Sanitization & Extraction  │
                    │   - Text Normalization (Unicode NFC)   │
                    │   - Boilerplate & Whitespace Stripping │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Chunking & Metadata Enrichment         │
                    │   - Ingestion Timestamp                │
                    │   - Tenant ID / RBAC Access Flags      │
                    │   - Document Version & Source URI      │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Vector Batch Embedding & Store Update  │
                    │   - ChromaDB Collection Upsert         │
                    │   - SQLite Metadata Master Record Sync │
                    └────────────────────────────────────────┘
```

---

## 2. Theoretical Mechanics: Change Data Capture & Idempotency

### SHA-256 Content Fingerprinting
To prevent costly redundant re-embedding of multi-megabyte documents, the pipeline calculates a deterministic cryptographic digest:

$$\mathcal{H}(D) = \text{SHA-256}(\text{RawContent} \parallel \text{SchemaVersion} \parallel \text{TenantID})$$

If $\mathcal{H}(D) \in \mathcal{K}_{\text{indexed}}$, the ingestion worker aborts immediately, guaranteeing $O(1)$ idempotency verification before any parsing or embedding compute is expended.

### Tombstone & Pruning Mechanics
When a document is deleted or modified at the source:
1. **Soft Delete (Tombstoning)**: Metadata attribute `is_active = false` is written to ChromaDB chunk metadata.
2. **Hard Pruning**: Asynchronous garbage collection sweeps delete chunk IDs where `doc_id = target_doc_id`, freeing HNSW memory.

---

## 3. Ingestion Architectural Trade-Offs

| Ingestion Strategy | Throughput | Freshness / Latency | Compute Resource Load | Failure Recovery |
|---|---|---|---|---|
| **Batch Periodic ETL** (Nightly Cron) | High ($> 10\text{k docs/min}$) | Low (Stale up to $24\text{h}$) | Spiky (High peak RAM/VRAM) | Simple retry from checkpoint |
| **Micro-Batch Queue** (Kafka / RabbitMQ) | High ($> 5\text{k docs/min}$) | Medium ($10\text{s} - 60\text{s}$) | Smooth & throttled | Dead-letter queue (DLQ) replay |
| **Real-Time Streaming** (CDC Webhooks) | Low ($< 500\text{ docs/min}$) | High ($< 1\text{s}$) | Continuous low CPU | Complex distributed rollback |

---

## 4. Failure Modes & Edge Cases

1. **Embedding API Rate Limit / 429 Exhaustion**:
   - *Failure*: Ingesting 50,000 chunks simultaneously overwhelms remote embedding endpoints.
   - *Mitigation*: Implement token-bucket rate limiters with exponential backoff and jitter ($t_{\text{wait}} = 2^k \cdot \text{base} + \text{rand}(0, 1)$).
2. **Ghost Chunks After Document Updates**:
   - *Failure*: Document A (originally 10 chunks) is edited to be shorter (4 chunks). Ingesting the new version leaves 6 orphaned chunks in ChromaDB.
   - *Mitigation*: Delete all existing chunks associated with `doc_id` before inserting newly generated chunks within an atomic transaction.

---

## 5. SOLID Principles in Document Ingestion

- **Single Responsibility (SRP)**: `DocumentReader` extracts bytes; `ContentHasher` produces hashes; `IngestionOrchestrator` schedules pipeline stages.
- **Open/Closed (OCP)**: New storage backends (S3, Azure Blob, Local FS) implement `SourceConnectorProtocol` without modifying parser or embedding layers.
- **Liskov Substitution (LSP)**: Any connector implementing `SourceConnectorProtocol` yields standardized `RawDocument` objects.
- **Interface Segregation (ISP)**: Separate `FileReaderProtocol` from `FileDeleterProtocol`.
- **Dependency Inversion (DIP)**: `IngestionPipeline` depends on `EmbedderProtocol` and `VectorStoreProtocol`.

---

## Implementation

The ingestion service is implemented in [ingestion_manager.py](implementation/ingestion_manager.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Zero duplicate embedding generation, constant-time deduplication lookups.
- **Weaknesses**: Requires persistent hash state storage for enterprise scale.
- **Trade-offs**: Memory overhead of state store vs re-embedding compute cost.

## Failure Modes & Mitigations

- **Near-duplicate docs with minute formatting variations**: Mitigated by text normalization prior to hashing or locality-sensitive hashing (MinHash).

## Architectural Decision & Trade-off Questions

1. *How do you handle real-time vs batch ingestion when syncing millions of enterprise documents to a vector store?*
2. *What is the impact of document deletion and how do you achieve tombstoning and sync in ChromaDB?*
