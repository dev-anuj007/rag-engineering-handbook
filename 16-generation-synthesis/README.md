# 16 Generation & Synthesis

## Objective

Master LlamaIndex response synthesizers (Compact, Tree Summarize, Refine), streaming generation, structured citation grounding, and strict refusal mechanisms.

## Why It Matters

Naive string concatenation of 10 chunks into an LLM prompt leads to fragmented or rambling responses. Structured synthesizers generate cohesive, fully-cited answers with zero context hallucination.

## Core Concepts

- **Compact & Refine**: Packs multiple chunks per prompt to minimize LLM roundtrips; refines sequentially if context spans multiple windows.
- **Tree Summarize**: Recursively builds hierarchical summary trees over hundreds of chunks for high-level thematic queries.
- **Strict Grounding Refusal**: Programmatically refuses when retrieved nodes contain zero supporting evidence.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of Attributed Synthesis

Generation synthesis is the bounded decoding of an autoregressive Large Language Model conditioned on both the user query and an isolated, tamper-evident evidence block.

```
                    ┌────────────────────────────────────────┐
                    │ System Instructions & Grounding Rules  │
                    │ ("Answer strictly using context only") │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Evidence Blocks with Explicit IDs      │
                    │   <doc id="1">Text A...</doc>          │
                    │   <doc id="2">Text B...</doc>          │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ User Question                          │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ LLM Autoregressive Generation          │
                    │   - Temperature: T = 0.0 (Greedy)      │
                    │   - Inline Citation Tags: [doc:1]      │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Post-Generation Attribution Verifier   │
                    │   - Verify all [doc:N] tags exist      │
                    │   - Assert claim entailment            │
                    └────────────────────────────────────────┘
```

---

## 2. Mathematical Formalization: Temperature, Sampling & Entailment

### 1. Temperature Scaling & Determinism:
For a vocabulary logit vector $\mathbf{z} \in \mathbb{R}^{|\mathcal{V}|}$, the generation probability distribution is:

$$P(y_t = v \mid y_{<t}, x) = \frac{\exp(z_v / T)}{\sum_{j \in \mathcal{V}} \exp(z_j / T)}$$

- In factual enterprise RAG, set **$T \rightarrow 0$ (Greedy Decoding / Argmax)** to eliminate stochastic drift and ensure reproducible outputs.

### 2. Natural Language Inference (NLI) Claim Entailment:
Every generated atomic sentence $s \in y$ must satisfy directional logical entailment against retrieved context $\mathcal{C}$:

$$\text{Entailment}(s, \mathcal{C}) = \begin{cases} 1 & \text{if } P_{\text{NLI}}(\text{Entailment} \mid \text{premise}=\mathcal{C}, \text{hypothesis}=s) \ge \tau_{\text{faith}} \\ 0 & \text{otherwise (Flagged as Hallucination)} \end{cases}$$

---

## 3. Synthesis Mode Trade-Off Matrix

| Synthesis Mode | Latency (TTFT) | Citation Precision | Hallucination Risk | Cost |
|---|---|---|---|---|
| **Direct Prompt Grounding** | $< 300\text{ ms}$ | Moderate | Low (with $T=0$) | Low ($1\times$ tokens) |
| **Inline Citation Generation** | $< 350\text{ ms}$ | High ($[doc:1]$ tags) | Very Low | Low ($1.1\times$ tokens) |
| **Refine / Iterative Accumulation** | $> 1200\text{ ms}$ | High | Extremely Low | High ($N\times$ tokens) |
| **Tree Summarization (Hierarchical)**| $> 2000\text{ ms}$ | Moderate | Low | Very High ($K\times$ tokens) |

---

## 4. Failure Modes & Mitigations

1. **Hallucinated Citation References**:
   - *Failure*: The model generates claims with invalid citation markers (e.g. `[doc:99]` when only docs 1–3 were provided).
   - *Mitigation*: Run a regex validation pass post-generation; strip non-existent citation tags or reject generation if citations fail verification.
2. **Refusal to Answer on Partial Matches**:
   - *Failure*: The context has $80\%$ of the necessary information, but over-aggressive grounding rules cause the LLM to completely refuse to answer.
   - *Mitigation*: Instruct the model to clearly delineate known facts from missing parameters (*"Based on the provided records, Acme is active in Texas; however, revenue figures for 2024 were not specified in the documents."*).

---

## 5. SOLID Principles in Generation & Synthesis

- **Single Responsibility (SRP)**: `PromptFormatter` structures XML tags; `LLMClient` executes inference; `CitationValidator` parses citation references.
- **Open/Closed (OCP)**: Different foundation models (Gemini 1.5 Pro, Claude 3.5 Sonnet, Llama 3) implement `LLMGeneratorProtocol`.
- **Liskov Substitution (LSP)**: All generators return structured `SynthesisResult(answer: str, citations: list[str], tokens_used: int)`.
- **Dependency Inversion (DIP)**: Synthesizer depends on `LLMGeneratorProtocol`.

---

## Implementation

Implemented in [synthesizer.py](implementation/synthesizer.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Compact**: Fastest latency and lowest token cost for standard QA.
- **Tree Summarize**: Capable of summarizing entire document collections at the expense of multiple LLM calls.

## Architectural Decision & Trade-off Questions

1. *How do you prevent response synthesis latency degradation when dealing with 50+ retrieved nodes?*
2. *How do you enforce inline bracketed citations `[Doc 1, p. 4]` during generation without degrading LLM fluency?*
