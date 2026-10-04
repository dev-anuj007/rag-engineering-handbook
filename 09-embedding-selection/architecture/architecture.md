# Embedding Selection Framework: Theoretical Deep Dive

## 1. First-Principles Selection Optimization

Selecting an embedding model is a multi-objective Pareto optimization problem balancing **MTEB retrieval accuracy**, **inference latency**, **vector RAM footprint**, and **operational inference cost**.

```
                           [ Pareto Frontier Optimization ]
  MTEB Retrieval Score ▲
                       │                     Frontier Optimum (BGE-Large / Gemini)
                       │                              *
                       │                    * (BGE-Base)
                       │             * (all-MiniLM-L6)
                       │
                       │       * (Sub-optimal Model)
                       │
                       └────────────────────────────────────────► Embedding Latency / Cost
```

---

## 2. Mathematical Capacity & Hardware Sizing Formulas

### 1. Raw Vector Memory Footprint:

$$\text{RAM}_{\text{raw}} = N_{\text{vectors}} \times D \times B_{\text{dtype}}$$

Where:
- $N_{\text{vectors}}$: Total number of indexed chunks (e.g. $10^7$).
- $D$: Vector dimension (e.g. $1536$).
- $B_{\text{dtype}}$: Bytes per dimension ($4\text{ bytes for float32}$, $2\text{ bytes for float16}$, $1\text{ byte for int8}$).

### 2. Matryoshka Representation Learning (MRL) Truncation:
Models trained with MRL (e.g. `text-embedding-3-large`, `nomic-embed-text`) permit slicing the first $D_{\text{sub}}$ dimensions:

$$\mathbf{v}_{\text{sub}} = \frac{\mathbf{v}_{1:D_{\text{sub}}}}{\|\mathbf{v}_{1:D_{\text{sub}}}\|_2}$$

Truncating from $D=3072 \rightarrow 512$ reduces RAM by $\mathbf{6\times}$ while retaining $>98\%$ of full-dimensional retrieval accuracy.

---

## 3. Decision Matrix for Enterprise Embedding Selection

| Workload Requirement | Recommended Embedding Class | Target Dimension ($D$) | Expected p95 Search Latency |
|---|---|---|---|
| **Ultra-Low Latency Edge / Mobile** | Small Quantized Bi-Encoder (`all-MiniLM-L6-v2`) | $384\text{d}$ | $< 5\text{ ms}$ |
| **Enterprise Standard Search** | Mid-Tier Bi-Encoder (`bge-base-en-v1.5`) | $768\text{d}$ | $< 15\text{ ms}$ |
| **Legal / Medical High Accuracy** | Frontier API / MRL (`text-embedding-004` / `bge-large`) | $1536\text{d}$ (or MRL 768) | $< 35\text{ ms}$ |
| **Code Search & AST Linking** | Specialized Code Bi-Encoder (`unixcoder` / `voyage-code`) | $1024\text{d}$ | $< 25\text{ ms}$ |

---

## 4. Failure Modes & Mitigations

1. **Premature Dimension Over-Provisioning**:
   - *Failure*: Selecting a 3072-dimension model for a 10-million document catalog requiring 120GB+ RAM, leading to massive cloud infrastructure costs.
   - *Mitigation*: Benchmark MRL truncation at 512d and 768d on internal evaluation sets before deploying full dimension.
2. **Context Window Overflow at Embedding**:
   - *Failure*: Ingestion pipeline silently truncates chunks exceeding the embedding model's 512-token limit, discarding trailing key information.
   - *Mitigation*: Enforce chunking limits with model-specific tokenizers matching the embedding tokenizer.

---

## 5. SOLID Principles in Model Selection

- **Single Responsibility (SRP)**: `ModelProfile` holds hardware metrics; `SelectionEngine` computes Pareto scoring.
- **Open/Closed (OCP)**: New embedding models register via `ModelRegistry` without rewriting scoring math.
- **Dependency Inversion (DIP)**: Pipeline components query `EmbeddingProfileProvider` to determine chunk sizing constraints.
