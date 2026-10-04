# 07 Embedding Fundamentals

## Objective

Master vector space geometry, high-dimensional dense representations, distance metrics (Cosine vs L2 vs Dot Product), and their native indexing in ChromaDB collections.

## Why It Matters

Mismatched distance metrics (e.g. indexing with L2 but querying assuming Cosine) produce inverted or erratic retrieval rankings, destroying RAG relevance.

## Core Concepts

- **High-Dimensional Geometry**: Dense float32/float16 embeddings map discrete semantic concepts into continuous manifold topologies.
- **Metric Selection**: Cosine similarity measures orientation independent of length; Dot product rewards magnitude when vectors are unnormalized.
- **ChromaDB HNSW Metric Configuration**: Specifying `hnsw:space: cosine` at collection creation.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of Text Embeddings

Text embedding models map discrete categorical tokens $\mathcal{V}^*$ into continuous, dense geometric vector representations in a high-dimensional Riemannian manifold $\mathbb{R}^D$ ($D \in [384, 3072]$) such that semantic similarity corresponds to geometric proximity.

```
Discrete Text Space                          Dense Embedding Manifold (R^D)
┌───────────────────────┐                    ┌────────────────────────────┐
│ "Refund Policy"       │ ──► [Bi-Encoder] ──► │ v1 = [ 0.23, -0.45, 0.81 ] │
│ "How to return items" │ ──► [Bi-Encoder] ──► │ v2 = [ 0.21, -0.43, 0.84 ] │ ◄── High Cosine Sim (0.98)
│ "Deep learning GPU"   │ ──► [Bi-Encoder] ──► │ v3 = [-0.67,  0.88, 0.12 ] │ ◄── Low Cosine Sim (0.05)
└───────────────────────┘                    └────────────────────────────┘
```

---

## 2. Mathematical Formalization of Distance Metrics

Given two dense vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^D$:

### 1. Cosine Similarity ($[-1, 1]$):

$$\text{CosineSim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^D u_i v_i}{\sqrt{\sum_{i=1}^D u_i^2} \sqrt{\sum_{i=1}^D v_i^2}}$$

### 2. Dot Product (Inner Product):

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v} = \sum_{i=1}^D u_i v_i$$

$$\text{Theorem: If } \|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1 \text{ (Unit $L_2$ Normalization), then } \text{CosineSim}(\mathbf{u}, \mathbf{v}) \equiv \langle \mathbf{u}, \mathbf{v} \rangle$$

### 3. Euclidean ($L_2$) Distance:

$$\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^D (u_i - v_i)^2}$$

$$\text{For unit-normalized vectors: } \|\mathbf{u} - \mathbf{v}\|_2^2 = 2 - 2 \cdot \text{CosineSim}(\mathbf{u}, \mathbf{v})$$

```
          [ Vector Normalization & Geometry ]
                      +y
                       │
                 v1    │   v2 (Cosine Sim = cos θ)
                  \    │   /
                   \   │  /
                    \  │ /
                     \ θ/
                      \│/
         ──────────────┼────────────── +x
                       │
                       │
                       │ Unit Circle: ||v|| = 1
```

---

## 3. Embedding Distance Metric Trade-Offs

| Metric | Computation Cost | Requires $L_2$ Normalization | Index Type in ChromaDB | Best Use Case |
|---|---|---|---|---|
| **Cosine Distance** | Moderate ($D$ mults + 2 norms) | No (computed dynamically) | `cosine` | Unnormalized vectors with varying text lengths |
| **Dot Product** | Fastest ($D$ multiply-adds) | Yes (to prevent magnitude distortion) | `ip` (inner product) | Unit-normalized vectors for ultra-low latency |
| **Euclidean ($L_2$)** | Moderate ($D$ subtract-squares) | Optional | `l2` | Geometric spatial clustering & outlier detection |

---

## 4. Failure Modes & Mitigations

1. **Unnormalized Dot Product Skew**:
   - *Failure*: Long chunks with large vector magnitudes dominate search results regardless of semantic relevance.
   - *Mitigation*: Enforce unit $L_2$ normalization ($\mathbf{v} \leftarrow \mathbf{v} / \|\mathbf{v}\|_2$) at ingestion before indexing.
2. **Curse of Dimensionality ($D > 1536$)**:
   - *Failure*: In ultra-high dimensional spaces, the ratio of distance to the nearest neighbor vs furthest neighbor approaches 1, degrading ANN partition efficiency.
   - *Mitigation*: Apply Matryoshka Representation Learning (MRL) truncation or PCA dimensionality reduction.

---

## 5. SOLID Principles in Embedding Services

- **Single Responsibility (SRP)**: `VectorNormalizer` handles $L_2$ scaling; `EmbeddingService` coordinates tensor inference.
- **Open/Closed (OCP)**: New embedding providers (Gemini, HuggingFace, OpenAI) implement `EmbeddingServiceProtocol`.
- **Liskov Substitution (LSP)**: All embedding engines return `list[float]` of length $D$.
- **Dependency Inversion (DIP)**: Vector stores depend on `EmbeddingServiceProtocol` for query vector generation.

---

## Implementation

Implemented in [vector_math.py](implementation/vector_math.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Cosine**: Invariant to document length variations.
- **Dot Product**: Sub-millisecond compute via SIMD hardware acceleration when normalized.

## Architectural Decision & Trade-off Questions

1. *Why does L2 distance equal $2(1 - \text{Cosine})$ when embeddings are unit-normalized ($L_2 = 1$)?*
2. *What is the curse of dimensionality and how does it degrade distance discrimination in $>1536$ dimensions?*
