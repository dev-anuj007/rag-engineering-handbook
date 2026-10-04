# 15 Context Engineering

## Objective

Optimize prompt packing, mitigate the "Lost in the Middle" attention degradation curve, enforce strict token budgeting, and inject explicit citation metadata.

## Why It Matters

LLMs exhibit high positional bias: information placed in the middle of long multi-chunk prompts is frequently ignored or hallucinated. Reordering context strategically restores attention focus on top-ranked evidence.

## Core Concepts

- **Positional Attention Bias**: LLMs recall evidence best when placed at the exact start or end of context.
- **Lost-in-the-Middle Reordering**: Alternating placement of high-ranking nodes to prompt extremes.
- **Hard Token Budgeting**: Preventing prompt overflow and truncating noisy tail nodes.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of Context Engineering

Context engineering is the science of structuring, filtering, compressing, and ordering retrieved evidence within the LLM prompt to maximize synthesis accuracy, eliminate redundant tokens, and counteract positional attention degradation.

```
                    Retrieved Top-K Candidates (from Reranker)
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │  Deduplication & Similarity Filtering  │
                    │   - Cosine threshold: discard duplicates│
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Context Compression & Token Pruning    │
                    │   - Extractive sentence summarization  │
                    │   - LLMLingua perplexity token pruning │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ "Lost-in-the-Middle" Position Ordering │
                    │   - Rank 1: Top (Prompt Prefix)        │
                    │   - Rank 2: Bottom (Prompt Suffix)     │
                    │   - Rank 3+: Alternating Center        │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Structured XML / Markdown Prompt Block │
                    └────────────────────────────────────────┘
```

---

## 2. Mathematical Formalization: The Lost-in-the-Middle Attention Curve

Empirical research (Liu et al., 2023) demonstrates that transformer multi-head self-attention exhibits a **U-shaped recall curve** with respect to token positions:

$$\text{Attention Recall}(p) \propto \alpha \cdot \exp(-\beta \cdot p) + \gamma \cdot \exp(-\delta \cdot (1 - p))$$

Where $p \in [0, 1]$ represents the relative normalized position in the prompt context window.

```
LLM Recall
Accuracy (%) ▲
      100%   │  * (Pos 0%: High Recall)               * (Pos 100%: High Recall)
             │   \                                   /
             │    \                                 /
             │     \                               /
       30%   │      \                             /
             │       \                           /
             │        \_________________________/  (Pos 40-60%: "Lost in the Middle")
             └────────────────────────────────────────────────► Normalized Position (p)
               0.0 (Start)          0.5 (Middle)        1.0 (End)
```

### Alternating Peripheral Reordering Algorithm:
To mitigate this curve, sort reranked documents $\mathcal{D} = [d_1, d_2, \dots, d_K]$ such that highest-ranked documents are positioned at the extremes:

$$\text{Layout}(\mathcal{D}) = [d_1, d_3, d_5, \dots, d_6, d_4, d_2]$$

---

## 3. Context Optimization Techniques Comparison

| Technique | Compression Ratio | Latency Overhead | Preservation of Numeric Facts |
|---|---|---|---|
| **Raw Concatenation** | $1.0\times$ (0% compression) | $0\text{ ms}$ | $100\%$ |
| **Information Extraction / LLM Compression** | $3.0\times - 5.0\times$ | High ($\sim 300\text{ ms}$) | Moderate ($85\%$) |
| **Perplexity Token Pruning (LLMLingua)** | $2.0\times - 3.0\times$ | Low ($\sim 25\text{ ms}$) | High ($92\%$) |
| **Peripheral Reordering (Lost-in-Middle)** | $1.0\times$ | $0\text{ ms}$ | $\mathbf{100\%}$ |

---

## 4. Failure Modes & Mitigations

1. **Context Window Token Saturation**:
   - *Failure*: Ingesting 10 large documents pushes prompt length past LLM limits or dramatically increases cost.
   - *Mitigation*: Enforce a strict context token budget (e.g. $4000\text{ tokens}$), terminating context insertion once the cumulative budget is exhausted.
2. **Loss of Table Structure During Extraction**:
   - *Failure*: Sentence-level compression extracts isolated numbers from a table, destroying column relationships.
   - *Mitigation*: Mark structured blocks with metadata flags `do_not_compress=True` to bypass lossy compressors.

---

## 5. SOLID Principles in Context Engineering

- **Single Responsibility (SRP)**: `ContextCompressor` prunes tokens; `PeripheralReorderer` organizes position layout; `PromptBuilder` formats XML blocks.
- **Open/Closed (OCP)**: Compression algorithms implement `CompressorProtocol`.
- **Liskov Substitution (LSP)**: All compressors accept and return `list[Document]`.
- **Dependency Inversion (DIP)**: `ContextBuilder` accepts `CompressorProtocol` via constructor injection.

---

## Implementation

Implemented in [context_compressor.py](implementation/context_compressor.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Noticeable reduction in context omission errors without additional model fine-tuning.
- **Weaknesses**: Requires deterministic character/token budgeting calculations.

## Architectural Decision & Trade-off Questions

1. *How does prompt packing order affect reasoning performance in long-context models (e.g. 1M token contexts)?*
2. *How do you dynamically allocate prompt budget between system instructions, few-shot examples, retrieved context, and conversation history?*
