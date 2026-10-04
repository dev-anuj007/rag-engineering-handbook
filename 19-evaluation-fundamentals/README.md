# 19 Evaluation Fundamentals

## Objective

Establish rigorous, scientific evaluation methodology for RAG systems, isolating retrieval failures from generation failures via component-wise benchmarks stored in SQLite.

## Why It Matters

Evaluating only end-to-end user answers conceals whether an incorrect response stemmed from missing retrieved context (retrieval failure) or an unfaithful LLM output (hallucination).

## Core Concepts

- **Evaluation Triad**: Evaluating Context Relevance, Groundedness (Faithfulness), and Answer Relevance.
- **Component-Wise Isolation**: Benchmarking the retrieval engine independently from LLM synthesis.
- **SQLite Evaluation Bench**: Persistent regression suites tracking metric regressions across git commits.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of RAG Evaluation

Evaluating a RAG system requires decoupling the **Retrieval Subsystem** (Did we find the right evidence?) from the **Generation Subsystem** (Did the model synthesize a truthful, grounded answer from that evidence?).

```
                                  [ THE RAG TRIAD EVALUATION MODEL ]

                                        ┌─────────────────┐
                                        │ User Query (Q)  │
                                        └────────┬────────┘
                                                 │
                             ┌───────────────────┴───────────────────┐
                             │                                       │
                1. Context Relevance                                 │ 3. Answer Relevance
               (Is C relevant to Q?)                                 │ (Does A answer Q?)
                             │                                       │
                             ▼                                       ▼
                  ┌──────────────────────┐               ┌──────────────────────┐
                  │ Retrieved Context (C)│ ────────────► │ Generated Answer (A) │
                  └──────────────────────┘ 2. Groundedness/└──────────────────────┘
                                              Faithfulness
                                         (Is A supported by C?)
```

---

## 2. Mathematical Formalization: The RAG Triad Metrics

Given query $Q$, retrieved context chunks $C = \{c_1, \dots, c_k\}$, and generated answer $A$:

### 1. Context Relevance:

$$\text{Score}_{\text{ctx\_rel}}(Q, C) = \frac{|\{c_i \in C \mid \text{Entails}(Q \rightarrow c_i)\}|}{|C|}$$

### 2. Faithfulness / Groundedness (Hallucination Inversion):
Let $A = \{s_1, s_2, \dots, s_m\}$ be the atomic claims extracted from the answer.

$$\text{Faithfulness}(A, C) = \frac{\sum_{j=1}^m \mathbb{I}(\text{SupportedBy}(s_j, C))}{m}$$

$$\text{Hallucination Rate} = 1.0 - \text{Faithfulness}(A, C)$$

### 3. Answer Relevance:
Measures semantic similarity between generated answer $A$ and the original intent of $Q$, penalizing evasions or off-topic hallucinations.

---

## 3. Evaluation Paradigm Trade-Off Matrix

| Evaluation Method | Automation Level | Latency per Sample | Cost per 1K Samples | Human Alignment |
|---|---|---|---|---|
| **Deterministic String / Regex** | 100% Automated | $< 1\text{ ms}$ | $\$0.00$ | Very Poor (Brittle to phrasing) |
| **Statistical (ROUGE / BLEU / BERTScore)** | 100% Automated | $< 10\text{ ms}$ | $\$0.00$ | Moderate ($0.50 - 0.65$) |
| **LLM-as-a-Judge (G-Eval / RAGAS)** | 100% Automated | $\sim 500\text{ ms}$ | $\$1.00 - \$5.00$ | Very High ($0.85 - 0.92$) |
| **Human Expert Review** | Manual | Hours / Days | $\$500.00+$ | Gold Standard ($1.00$) |

---

## 4. Failure Modes & Mitigations

1. **Conflating Retrieval and Generation Failures**:
   - *Failure*: An incorrect answer is logged as an LLM hallucination, but inspection reveals the retriever failed to find the source chunk (Recall failure).
   - *Mitigation*: Isolate offline evaluation into two distinct test phases: evaluate retrieval with Recall@K / MRR; evaluate generation with Faithfulness conditioned on golden context.
2. **Evaluation Data Contamination (Data Leakage)**:
   - *Failure*: Test queries are included in the ingestion training set, yielding artificially inflated 100% recall scores.
   - *Mitigation*: Maintain strict temporal and cryptographic train/test splits in SQLite.

---

## 5. SOLID Principles in Evaluation Fundamentals

- **Single Responsibility (SRP)**: `RetrievalEvaluator` scores context recall; `GenerationEvaluator` scores claim faithfulness; `MetricAggregator` computes dataset averages.
- **Open/Closed (OCP)**: Custom evaluation metrics implement `EvaluationMetricProtocol`.
- **Liskov Substitution (LSP)**: All metrics return standardized `MetricResult(name: str, score: float, details: dict)`.
- **Dependency Inversion (DIP)**: `EvaluationRunner` depends on `EvaluationMetricProtocol` collections.

---

## Implementation

Implemented in [metrics_engine.py](implementation/metrics_engine.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Pinpoints root causes of system degradation with surgical precision.
- **Weaknesses**: Requires curated golden datasets with ground truth context spans.

## Architectural Decision & Trade-off Questions

1. *How do you build an automated CI/CD regression test suite for a production RAG system?*
2. *What is the difference between Reference-Based evaluation (requires golden answers) and Reference-Free evaluation (LLM-as-a-judge on source context)?*
