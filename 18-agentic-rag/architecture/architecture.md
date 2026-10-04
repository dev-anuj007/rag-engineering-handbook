# Agentic RAG: Theoretical Deep Dive

## 1. First-Principles Mechanics of Agentic RAG

Agentic RAG transitions from static, single-step retrieval into an **autonomous multi-step reasoning loop (ReAct / Plan-and-Solve)** where the LLM dynamically decides *when* to search, *what* tools to call, and *how* to decompose complex queries across multiple heterogeneous knowledge silos.

```
                             [ ReAct AGENTIC REASONING LOOP ]

                                  User Complex Multi-Hop Query
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │    Thought / Reason   │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ Action / Tool Call:   │
                                    │ vector_search("Acme") │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ Observation:          │
                                    │ "Acme CEO is Alice"   │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ Thought / Reason:     │
                                    │ "Need Alice's tenure" │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ Action / Tool Call:   │
                                    │ sql_query("tenures")  │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ Final Synthesized Ans │
                                    └───────────────────────┘
```

---

## 2. Mathematical Formalization: ReAct State Space & Multi-Hop Traversal

At reasoning step $t \in \{1, \dots, T\}$:

1. **Thought State Generation**:

$$\tau_t \sim P_\theta(\tau_t \mid x, a_1, o_1, \dots, a_{t-1}, o_{t-1})$$

2. **Action / Tool Dispatch**:

$$a_t \sim P_\theta(a_t \mid x, a_1, o_1, \dots, \tau_t)$$

3. **Environment Observation**:

$$o_t = \text{ExecuteTool}(a_t) \quad (\text{e.g. Query ChromaDB or execute SQL})$$

4. **Termination Criterion**:
   The loop terminates when $a_t = \text{FINISH}(y)$ or when $t > T_{\text{max\_steps}}$.

---

## 3. Agentic vs Static Retrieval Trade-Offs

| Dimension | Static RAG (Single-Step) | Multi-Agent / ReAct RAG |
|---|---|---|
| **Multi-Hop Query Resolution** | Fails (Cannot synthesize across disjoint queries) | State-of-the-Art |
| **p95 Latency** | $\sim 500\text{ ms}$ | $\sim 3000\text{ ms} - 8000\text{ ms}$ |
| **Token Consumption** | Low ($1\times$) | High ($5\times - 15\times$) |
| **Tool Diversity** | Vector Search Only | Vector, SQL, Web API, Code Execution |

---

## 4. Failure Modes & Mitigations

1. **Reasoning Loop Traps & Tool Call Thrashing**:
   - *Failure*: Agent repeatedly generates slightly modified search queries that return empty results, burning tokens until reaching max context.
   - *Mitigation*: Track query similarity history; if cosine similarity between consecutive search actions $> 0.85$, force state transition or termination.
2. **Context Bleed in Long Multi-Turn Agent Trajectories**:
   - *Failure*: Massive tool observation strings saturate the agent's context window.
   - *Mitigation*: Summarize and truncate tool observations before appending to trajectory history.

---

## 5. SOLID Principles in Agentic RAG

- **Single Responsibility (SRP)**: `AgentEngine` maintains state history; `ToolRegistry` executes actions.
- **Open/Closed (OCP)**: New tools (ChromaDB, Calculator, Python REPL) implement `AgentToolProtocol`.
- **Liskov Substitution (LSP)**: All tools return standardized `ToolOutput(content: str, success: bool)`.
- **Dependency Inversion (DIP)**: Agent depends on `AgentToolProtocol` collections.
