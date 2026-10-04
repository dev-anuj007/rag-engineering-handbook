# Embedding Models: Theoretical Deep Dive

## 1. First-Principles Taxonomy of Embedding Models

Modern retrieval architectures leverage three distinct classes of embedding representations: Dense Dual-Tower Bi-Encoders, Learned Sparse Representations (SPLADE), and Multi-Vector ColBERT representations.

```
[ DENSE BI-ENCODER ]
Text ──► [Transformer Encoder] ──► Mean Pooling ──► Single Dense Vector (1x768)

[ LEARNED SPARSE (SPLADE) ]
Text ──► [Transformer Encoder] ──► MLM Logits + ReLU ──► Sparse Vocabulary Vector (1x30000)

[ MULTI-VECTOR (ColBERT) ]
Text ──► [Transformer Encoder] ──► Token Embeddings ──► Matrix of Token Vectors (Nx128)
```

---

## 2. Mathematical Mechanics

### 1. Dense Bi-Encoders
Maps passage $d$ to single vector $\mathbf{v}_d = \text{Pool}(\text{Encoder}(d))$. Similarity score is scalar dot product:

$$S_{\text{dense}}(q, d) = \mathbf{v}_q^\top \mathbf{v}_d$$

### 2. Learned Sparse Models (SPLADE)
Projects token representations to the entire vocabulary $\mathcal{V}$, yielding term weights $w_v$:

$$w_v(d) = \max_{t \in d} \log(1 + \text{ReLU}(W_v \mathbf{h}_t + b_v))$$

Scoring is an inverted index dot product over non-zero vocabulary weights:

$$S_{\text{sparse}}(q, d) = \sum_{v \in q \cap d} w_v(q) \cdot w_v(d)$$

### 3. Multi-Vector Late Interaction (ColBERT)
Computes token-to-token MaxSim operator across all query and document tokens:

$$S_{\text{ColBERT}}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{E}_{q, i}^\top \mathbf{E}_{d, j}$$

---

## 3. Embedding Paradigm Trade-Off Matrix

| Paradigm | Index Storage (1M Docs) | Query Latency (p95) | Out-of-Vocabulary / Keyword Recall | Semantic Reasoning Recall |
|---|---|---|---|---|
| **Dense Bi-Encoder** (e.g. BGE/Gemini) | $\sim 3\text{ GB}$ (768d float32) | $< 15\text{ ms}$ | Moderate ($0.65$) | Very High ($0.92$) |
| **Learned Sparse** (SPLADE) | $\sim 1.5\text{ GB}$ (Inverted index) | $< 25\text{ ms}$ | Very High ($0.94$) | High ($0.86$) |
| **Multi-Vector** (ColBERT v2) | $\sim 20\text{ GB}$ (Residual quantized) | $\sim 45\text{ ms}$ | Very High ($0.95$) | State-of-the-Art ($0.96$) |

---

## 4. Failure Modes & Mitigations

1. **ColBERT Index Size Bloat**:
   - *Failure*: Storing 128-dimensional vectors for every token across millions of documents exhausts server RAM.
   - *Mitigation*: Apply ColBERT v2 centroid residual quantization, compressing token vectors from 512 bytes to 16-32 bytes per token.
2. **Dense Semantic Bleed**:
   - *Failure*: Querying for exact part number `A102-X` returns unrelated electronic parts because the model assigns near-identical embeddings to structured part codes.
   - *Mitigation*: Hybridize with BM25 or SPLADE sparse vectors.

---

## 5. SOLID Principles in Embedding Modeling

- **Single Responsibility (SRP)**: `DenseEncoder` outputs fixed dense vectors; `SparseEncoder` yields sparse term-weight dictionaries.
- **Open/Closed (OCP)**: Multi-vector and Matryoshka models register via `GenericEncoderProtocol`.
- **Liskov Substitution (LSP)**: All single-vector encoders adhere to `VectorEncoderProtocol`.
- **Dependency Inversion (DIP)**: Indexing engine depends on `VectorEncoderProtocol`.
