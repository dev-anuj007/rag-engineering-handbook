# 21 Generation Evaluation

## Objective

Quantify the factual correctness, faithfulness (hallucination rate), and question relevance of LLM-generated synthesis against SQLite evaluation test suites.

## Why It Matters

An LLM can generate fluent, persuasive, but completely fabricated answers (hallucinations). Generation evaluation algorithms break answers into atomic claims and verify entailment against retrieved context.

## Core Concepts

- **Faithfulness (Groundedness)**: Proportion of claims in the generated response that can be mathematically or logically deduced from the provided context.
- **Answer Relevance**: Measure of whether the generated response directly answers the user's inquiry without extraneous digressions.
- **Hallucination Detection**: Automated alerting on ungrounded assertions.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of Generation Evaluation

Generation evaluation quantifies whether an LLM's synthesized response is **faithful to retrieved evidence** (groundedness), **relevant to the user query**, and **factually accurate** against reference answers.

```
                    Generated Answer (A)
                             │
                             ▼
             ┌───────────────────────────────┐
             │ Atomic Claim Decomposition    │
             │ Splits answer into sentences: │
             │ s1: "Acme founded in 2010"    │
             │ s2: "Revenue was $10M"        │
             └───────────────┬───────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │ Entailment Verification       │
             │ Checks s_i against Context C  │
             │ s1: ENTAILED (Supported)      │
             │ s2: NOT ENTAILED (Hallucinated│
             └───────────────┬───────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │ Faithfulness Score = 1 / 2    │
             │ Hallucination Rate = 50%      │
             └───────────────────────────────┘
```

---

## 2. Mathematical Formalization: Claim Entailment & Semantic Similarity

### 1. Atomic Claim Extraction:
Let answer $A$ decompose into $M$ independent propositions $\mathcal{P} = \{p_1, p_2, \dots, p_M\}$ using a structured extraction prompt or syntactic dependency parser.

### 2. Context Groundedness / Faithfulness:

$$\text{Faithfulness}(A, C) = \frac{\sum_{i=1}^M \mathbb{I}(C \models p_i)}{M}$$

Where $C \models p_i$ represents semantic entailment computed by Natural Language Inference (NLI) classifier or LLM Judge.

### 3. Answer Semantic Similarity (BERTScore / Embedding Cosine):

$$\text{Similarity}(A, A^*) = \cos(\mathbf{e}(A), \mathbf{e}(A^*)) = \frac{\mathbf{e}(A) \cdot \mathbf{e}(A^*)}{\|\mathbf{e}(A)\| \|\mathbf{e}(A^*)\|}$$

Where $A^*$ is the reference ground truth answer.

---

## 3. Generation Metric Trade-Off Matrix

| Metric | Target Dimension | Reference Required | Computation Cost | Sensitivity to Hallucination |
|---|---|---|---|---|
| **Faithfulness / Groundedness** | Context Entailment | No (Only retrieved context) | Moderate (NLI / LLM) | Extremely High |
| **Answer Relevance** | Query Intent Alignment | No (Only user query) | Moderate (LLM Judge) | Moderate |
| **Semantic Correctness** | Factual Match to Gold | Yes (Golden reference answer) | Low (Embedding Cosine) | High |
| **ROUGE-L / BLEU** | Lexical $N$-gram Overlap | Yes (Golden reference answer) | Zero ($< 1\text{ ms}$) | Very Poor |

---

## 4. Failure Modes & Mitigations

1. **Semantic Inversion in N-Gram Metrics (ROUGE / BLEU)**:
   - *Failure*: Generated answer *"Company was not profitable"* scores $90\%$ ROUGE against reference *"Company was profitable"*, despite being opposite in meaning.
   - *Mitigation*: Replace ROUGE/BLEU with NLI-based entailment scoring or LLM-as-a-Judge semantic checks.
2. **Compound Sentence Hallucination Masking**:
   - *Failure*: A sentence containing three true facts and one fabricated number is scored as fully correct if evaluated as a single unit.
   - *Mitigation*: Deconstruct sentences into minimal atomic single-fact propositions before entailment scoring.

---

## 5. SOLID Principles in Generation Evaluation

- **Single Responsibility (SRP)**: `ClaimExtractor` splits propositions; `EntailmentChecker` verifies support; `MetricAggregator` calculates ratios.
- **Open/Closed (OCP)**: New metrics (G-Eval, Faithfulness, ROUGE) implement `GenerationMetricProtocol`.
- **Liskov Substitution (LSP)**: All metrics return `GenerationScore(name: str, value: float, details: dict)`.
- **Dependency Inversion (DIP)**: Evaluation harness depends on `GenerationMetricProtocol`.

---

## Implementation

Implemented in [generation_metrics.py](implementation/generation_metrics.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Detects subtle parametric hallucinations missed by string matching.
- **Weaknesses**: Claim extraction with LLMs incurs additional inference cost during test runs.

## Architectural Decision & Trade-off Questions

1. *How do you evaluate generation quality when no ground truth golden answers exist in production?*
2. *How do you isolate whether a hallucination was caused by conflicting retrieved context vs model parametric memory drift?*
