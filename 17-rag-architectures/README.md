# 17 RAG Architectures

## Objective

Master advanced RAG architectural paradigms: Corrective RAG (CRAG), Self-RAG (self-reflection and hallucination grading), Modular RAG, and Adaptive RAG.

## Why It Matters

Standard linear RAG operates blindly: it executes retrieval and forces generation regardless of context quality. Corrective RAG dynamically self-grades retrieved nodes and triggers fallbacks or query rewriting when retrieval fails.

## Core Concepts

- **Corrective RAG (CRAG)**: Evaluates retrieval quality dynamically and branches between internal synthesis, ambiguous warning flags, or external web search.
- **Self-RAG**: Uses reflection critique tokens `[RETRIEVE]`, `[IS_REL]`, `[IS_SUP]` to self-correct during generation.
- **Modular RAG**: Decoupled, interchangeable routing, indexing, and synthesis micro-modules.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Taxonomy of RAG Paradigms

RAG architectures have evolved through four distinct generations: **Naive RAG**, **Advanced RAG**, **Modular RAG**, and **Dynamic / Self-Reflective RAG (CRAG & Self-RAG)**.

```
[ MODULAR RAG ROUTING TOPOLOGY ]

                             User Query
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Query Router / Intent │
                     └───────────┬───────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
┌─────────────┐           ┌─────────────┐             ┌─────────────┐
│ Fast Cache  │           │ Vector RAG  │             │ Complex SQL │
│ (Exact/Sem) │           │ (ChromaDB)  │             │ / Text-to-DB│
└─────────────┘           └─────────────┘             └─────────────┘
```

---

## 2. Mathematical Formalization of Self-RAG & Corrective RAG (CRAG)

### 1. Self-RAG: Special Reflection Tokens
Self-RAG (Asai et al.) trains the generator to dynamically output discrete critique tokens:
- **`[Retrieve]`**: Threshold $\tau_{\text{ret}} \in [0, 1]$. If $P(\text{Retrieve}=\text{YES} \mid x) > \tau_{\text{ret}}$, trigger vector retrieval.
- **`[IsRel]`**: Relevance of retrieved passage $d$: evaluates $P(\text{IsRel}=\text{Relevant} \mid x, d)$.
- **`[IsSup]`**: Groundedness support: evaluates whether generation $y$ is supported by context $d$.
- **`[IsUse]`**: Utility score: evaluates overall helpfulness of $y$ to query $x$.

### 2. Corrective RAG (CRAG) Retrieval Evaluator:
CRAG evaluates the confidence score $\mathcal{S}_{\text{rel}}$ of retrieved documents:

$$\text{Action} = \begin{cases} \text{Correct (Synthesize Directly)} & \text{if } \mathcal{S}_{\text{rel}} \ge \theta_{\text{high}} \\ \text{Ambiguous (Combine Vector + Web Search)} & \text{if } \theta_{\text{low}} \le \mathcal{S}_{\text{rel}} < \theta_{\text{high}} \\ \text{Incorrect (Discard & Fallback to Web Search)} & \text{if } \mathcal{S}_{\text{rel}} < \theta_{\text{low}} \end{cases}$$

---

## 3. Comparative Taxonomy Matrix

| Architecture | Control Flow | Retrieval Trigger | Self-Correction | Latency Profile |
|---|---|---|---|---|
| **Naive RAG** | Linear: Retrieve -> Generate | Static (Every query) | None | $\sim 300\text{ ms}$ |
| **Advanced RAG** | Pre-retrieval + Post-retrieval Rerank | Static | None | $\sim 500\text{ ms}$ |
| **Modular RAG** | Dynamic routing to microservices | Conditional | Service fallback | $\sim 600\text{ ms}$ |
| **Corrective RAG (CRAG)** | Evaluator Gate -> Fallback Web | Static initial, dynamic branch | Yes (Document filtering) | $\sim 900\text{ ms}$ |
| **Self-RAG** | Reflection token decoding | Dynamic (On demand) | Yes (Token critique) | $\sim 1500\text{ ms}$ |

---

## 4. Failure Modes & Mitigations

1. **Router Misclassification in Modular RAG**:
   - *Failure*: An analytical query is routed to the vector FAQ database instead of the SQL engine, returning shallow matching text instead of accurate calculated metrics.
   - *Mitigation*: Fallback multi-router execution when classification confidence is $< 0.85$.
2. **Infinite Loops in Self-Reflective Loops**:
   - *Failure*: A query that has no answer in any knowledge source triggers repeated re-retrieval and re-generation loops indefinitely.
   - *Mitigation*: Enforce a strict iteration bound ($\text{MaxRetries} = 2$) with circuit breakers.

---

## 5. SOLID Principles in RAG Architectures

- **Single Responsibility (SRP)**: `QueryRouter` classifies intent; `CRAGEvaluator` checks confidence; `PipelineExecutor` runs the workflow.
- **Open/Closed (OCP)**: New architectural patterns (e.g. Adaptive RAG, Graph-RAG) implement `RAGWorkflowProtocol`.
- **Liskov Substitution (LSP)**: All workflows execute `async def run(query: str) -> RAGResponse`.
- **Dependency Inversion (DIP)**: `WorkflowCoordinator` depends on `RAGWorkflowProtocol`.

---

## Implementation

Implemented in [crag_pipeline.py](implementation/crag_pipeline.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Robust protection against hallucinations on out-of-domain queries.
- **Weaknesses**: Grader step introduces additional intermediate latency.

## Architectural Decision & Trade-off Questions

1. *How does Self-RAG differ from Corrective RAG in execution topology and model requirement?*
2. *How do you build an Adaptive RAG router that classifies query complexity to choose between No-RAG, Simple RAG, and Multi-Hop RAG?*
