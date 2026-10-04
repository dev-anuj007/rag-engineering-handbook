# 24 Evaluation Datasets

## Objective

Build, curate, and evolve robust synthetic and human-reviewed golden evaluation datasets with explicit SQLite persistence and train/validation/test partitions.

## Why It Matters

Manual annotation of 1,000 QA pairs across complex enterprise manuals is time-prohibitive. Synthetic generation combined with Evol-Instruct complexity mutation produces diverse multi-hop evaluation benches at near-zero cost.

## Core Concepts

- **Corpus-Grounded Synthetic Generation**: LLM-driven generation of questions conditioned on specific corpus nodes.
- **Evol-Instruct Mutation**: Evolving simple queries into complex multi-hop, counterfactual, or comparative questions.
- **Dataset Splitting & Leakage Prevention**: Enforcing strict document-level isolation between test and training sets.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Dataset Synthesis Mechanics

High-confidence evaluation requires a diverse **Golden Dataset** $\mathcal{D}_{\text{eval}} = \{(q_i, \mathcal{C}_i, a_i^*, \mathcal{M}_i)\}_{i=1}^N$ containing user queries ($q_i$), relevant ground truth context chunks ($\mathcal{C}_i$), reference answers ($a_i^*$), and metadata tags ($\mathcal{M}_i$).

```
                    Raw Document Corpus (e.g. 10,000 Chunks)
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │ Chunk Sampler & Topic Clustering     │
                    │   - Embed chunks & cluster (k-means) │
                    │   - Sample representative centroids  │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │ Multi-Style Synthetic QA Generator   │
                    │   - Simple Factoid Questions         │
                    │   - Multi-Hop Cross-Chunk Questions  │
                    │   - Conditional & Reasoning Questions│
                    │   - Negative / Out-of-Domain Queries │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │ Quality Filter & Self-Verification   │
                    │   - Can the question be answered?    │
                    │   - Discard low-quality / ambiguous  │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │ SQLite Golden Dataset Repository     │
                    └──────────────────────────────────────┘
```

---

## 2. Mathematical Formalization: Synthetic Question Evolution

To prevent dataset homogeneity, synthetic question generation applies **Evolutionary Prompting (Evol-Instruct / RAGAS Testset Generation)** to mutate a simple seed question $q_0$:

### 1. Multi-Hop Reasoning Evolution:
Combines two disjoint context passages $c_A$ and $c_B$:

$$q_{\text{multi-hop}} \sim P_{\text{evolve}}(q \mid c_A, c_B) \quad \text{such that answering requires information from both } c_A \text{ and } c_B$$

### 2. Constraint & Conditionality Evolution:
Adds negative or boundary conditions:

$$q_{\text{conditional}} = q_0 + \text{" under the condition that transaction occurred after 2021"}$$

### 3. Out-of-Domain / Negative Generation:
Generates plausible-sounding queries whose answers do not exist in the corpus:

$$q_{\text{negative}} \sim P_{\text{adversarial}}(q \mid c), \quad \text{Ground Truth Context} = \emptyset, \quad a^* = \text{"UNKNOWN"}$$

---

## 3. Dataset Construction Strategy Matrix

| Strategy | Speed & Scalability | Diversity & Edge Cases | Ground-Truth Fidelity | Cost |
|---|---|---|---|---|
| **Manual Human Annotation** | Very Slow (10 QA / hour) | Low (Human fatigue bias) | Gold Standard ($100\%$) | Very High ($\$5.00+/\text{sample}$) |
| **Production User Log Sampling** | Fast (Continuous capture) | Real-world distributions | Variable (Requires manual verification) | Low |
| **Synthetic LLM Generation** | Ultra Fast ($1000\text{ QA} / \text{min}$) | Extremely High (Evol-Instruct) | High ($90\%+$ with verification filter) | Low ($\$0.02/\text{sample}$) |
| **Hybrid (Synthetic + Human Review)**| Moderate | High | Gold Standard ($99\%$) | Medium |

---

## 4. Failure Modes & Mitigations

1. **Synthetic Answer Hallucination Contamination**:
   - *Failure*: An LLM generator creates a synthetic question with an answer derived from its internal parametric weights rather than the supplied chunk.
   - *Mitigation*: Run a reverse verification pass: prompt a separate LLM to answer the question using *only* the chunk; discard if the answer does not match the generated gold answer.
2. **Class Imbalance in Evaluation Datasets**:
   - *Failure*: Dataset contains $95\%$ simple single-fact queries and $5\%$ complex multi-hop queries, masking multi-hop retrieval degradation in production.
   - *Mitigation*: Enforce stratified sampling across query archetypes in SQLite schema.

---

## 5. SOLID Principles in Dataset Generation

- **Single Responsibility (SRP)**: `EvolSynthesizer` mutates question complexity; `QAValidator` checks answerability; `DatasetStore` persists to SQLite.
- **Open/Closed (OCP)**: New mutation strategies implement `QuestionEvolverProtocol`.
- **Liskov Substitution (LSP)**: All generators return standard `GoldenSample` objects.
- **Dependency Inversion (DIP)**: Benchmark runners load samples via `GoldenDatasetRepositoryProtocol`.

---

## Implementation

Implemented in [dataset_generator.py](implementation/dataset_generator.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Enables massive test coverage across all document sections automatically.
- **Weaknesses**: Synthetic questions can lack the conversational messiness and typos of real end-users.

## Architectural Decision & Trade-off Questions

1. *How do you prevent data contamination when using synthetic questions to evaluate a RAG pipeline?*
2. *What is the role of 'negative examples' (queries with no relevant answer in the corpus) in golden evaluation sets?*
