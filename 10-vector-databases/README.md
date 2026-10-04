# 10 Vector Databases

## Objective

Deeply understand vector database internals, HNSW approximate nearest neighbor graphs, parameter tuning ($M$, $efConstruction$, $efSearch$), and ChromaDB lifecycle management.

## Why It Matters

Linear vector scans scale at $O(N)$ with dataset size, causing catastrophic latency spikes beyond 50k vectors. HNSW graph indexing provides sub-linear $O(\log N)$ retrieval latency with $>95\%$ recall.

## Core Concepts

- **HNSW Skip-List Multilayer Graphs**: Multi-tier routing allows logarithmic convergence to nearest vector clusters.
- **Index Parameter Tuning**: Balancing $M$ and $ef$ for memory consumption vs query QPS.
- **ChromaDB CRUD & Metadata Indexing**: Real-time vector inserts, metadata-filtered searches, and targeted deletions.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of Approximate Nearest Neighbor (ANN)

Exact $K$-Nearest Neighbor ($K$-NN) search via brute-force linear scan requires $O(N \cdot D)$ distance computations per query—prohibitive when $N > 10^6$. Vector databases (such as ChromaDB) solve this using **Approximate Nearest Neighbor (ANN)** indexing graphs.

```
                      [ HNSW GRAPH INDEX TOPOLOGY ]

Layer 2 (Express):      Node A ───────────────────────────────► Node Z
                          │                                       │
Layer 1 (Medium):       Node A ────────► Node G ──────────────► Node Z
                          │                │                      │
Layer 0 (Dense Base):   Node A ──► Node B ─┴─► Node C ──► Node G ─┴─► Node Z
```

---

## 2. Mathematical Formalization of HNSW & IVF-PQ

### 1. Hierarchical Navigable Small World (HNSW)
HNSW builds a multi-layer graph with probability decay for layer assignment:

$$l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor$$

- **$M$ (Max Edges per Node)**: Governs graph connectivity ($M \in [16, 64]$).
- **$\text{efConstruction}$**: Size of dynamic candidate list during build ($[100, 400]$).
- **$\text{efSearch}$**: Size of dynamic candidate list during query time ($[50, 200]$).

$$\text{HNSW Search Complexity} = O(\log N)$$

$$\text{HNSW Memory Overhead} = N \times M \times 2 \times 4\text{ bytes (Graph pointers)}$$

$$\text{Total HNSW RAM} \approx \text{RAM}_{\text{raw}} \times 1.5$$

### 2. Inverted File with Product Quantization (IVF-PQ)
1. **IVF Voronoi Partitioning**: Partitions vector space into $C$ coarse centroids using $k$-means. Search queries probe only the nearest $n_{\text{probe}}$ clusters.
2. **Product Quantization (PQ)**: Decomposes $D$-dimensional vector into $m$ sub-vectors of dimension $D/m$, quantizing each to the nearest of $k^*$ sub-centroids (represented by an 8-bit byte index):

$$\mathbf{v} \in \mathbb{R}^D \xrightarrow{\text{PQ}} [b_1, b_2, \dots, b_m] \in \{0, \dots, 255\}^m$$

$$\text{Compression Ratio} = \frac{D \times 4\text{ bytes}}{m \times 1\text{ byte}} = \frac{4D}{m} \quad (\text{typically } 8\times - 32\times \text{ compression})$$

---

## 3. Vector Database Indexing Trade-Off Matrix

| Indexing Technique | Search Latency (p95) | Recall@10 | Memory (RAM) Usage | Index Build Time |
|---|---|---|---|---|
| **Flat (Brute Force $K$-NN)** | $O(N \cdot D)$ (Slow: $>500\text{ms}$) | $1.00$ ($100\%$ exact) | $1.0\times$ (Raw vectors) | Zero build time |
| **HNSW** (ChromaDB Default) | $O(\log N)$ (Fast: $<5\text{ms}$) | $0.95 - 0.99$ | $1.5\times - 2.0\times$ (High RAM) | Moderate |
| **IVF-Flat** | $O(n_{\text{probe}} \frac{N}{C})$ ($10\text{ms}$) | $0.88 - 0.94$ | $1.1\times$ | Fast (Centroid clustering) |
| **IVF-PQ** | $O(n_{\text{probe}} \frac{N}{C})$ ($15\text{ms}$) | $0.80 - 0.90$ | $0.1\times - 0.2\times$ (Ultra-low) | High (Codebook training) |

---

## 4. Failure Modes & Mitigations

1. **Out-of-Memory (OOM) Crash on Large HNSW Collections**:
   - *Failure*: Ingesting 20M 1536-dimensional vectors into memory-resident HNSW consumes 180GB+ RAM, crashing the server.
   - *Mitigation*: Switch to memory-mapped storage (e.g. ChromaDB on-disk DuckDB/SQLite + persistent HNSW binary) or use scalar quantization (SQ8 / int8).
2. **Index Recall Degradation from Under-Tuned $\text{efSearch}$**:
   - *Failure*: Setting $\text{efSearch} < 20$ results in recall dropping below $70\%$ during high-concurrency query spikes.
   - *Mitigation*: Dynamically adjust $\text{efSearch}$ based on query priority ($100$ for high accuracy, $30$ for low latency).

---

## 5. SOLID Principles in Vector Database Management

- **Single Responsibility (SRP)**: `IndexManager` configures HNSW parameters; `VectorStoreAdapter` handles CRUD queries.
- **Open/Closed (OCP)**: New vector backends (ChromaDB, Qdrant, Milvus) implement `VectorDatabaseProtocol`.
- **Liskov Substitution (LSP)**: All database adapters expose consistent async search semantics returning standardized `ScoredDocument` collections.
- **Dependency Inversion (DIP)**: Retrievers depend on `VectorDatabaseProtocol`.

---

## Implementation

Implemented in [chroma_db_store.py](implementation/chroma_db_store.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: True sub-10ms nearest neighbor search over millions of records.
- **Weaknesses**: Significant memory overhead (approx. 1.5–2x raw vector size for graph edges).

## Architectural Decision & Trade-off Questions

1. *How does HNSW compare with IVFFlat (Inverted File Index) in terms of indexing throughput, memory, and recall?*
2. *How do you handle live deletion and tombstoning in HNSW graphs without triggering costly global re-indexing?*
