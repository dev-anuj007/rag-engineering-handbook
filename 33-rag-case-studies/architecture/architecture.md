# RAG Domain Case Studies: Theoretical Deep Dive

## 1. First-Principles Domain Adaptation Blueprint

No single RAG configuration works across all enterprise domains. Domain adaptation requires tailoring **chunking granularity**, **embedding specialization**, **metadata filtering rules**, and **synthesis prompt constraints** to the specific factual dynamics of each vertical.

```
                                  [ DOMAIN ADAPTATION TAXONOMY ]

┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│       Legal RAG         │   │      Financial RAG      │   │      Codebase RAG       │
├─────────────────────────┤   ├─────────────────────────┤   ├─────────────────────────┤
│ • Verbatim clause match │   │ • Tabular preservation  │   │ • AST / Tree-sitter     │
│ • Parent-Child chunking │   │ • Temporal filtering    │   │ • Call-graph linking    │
│ • Zero hallucination    │   │ • Numerical grounding   │   │ • Syntax-aware chunking │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

---

## 2. Comparative Domain Engineering Matrix

| Domain Vertical | Primary Chunking Strategy | Retrieval Backbone | Critical Guardrail |
|---|---|---|---|
| **Legal Discovery** | Parent-Child (Hierarchical clause blocks) | Hybrid (BM25 + Dense) + Cross-Encoder | Strict negation checking & clause section tagging |
| **Financial 10-K Analysis** | Table-preserving Markdown chunker | Dense ANN + Time-Series Metadata Filter | Numerical cell verification & currency normalization |
| **Healthcare / Clinical** | Sentence-window with medical ontology tags | Specialized Medical Embedder (BioBERT/Med-Embed) | Confidence threshold refusal when evidence is incomplete |
| **Code Search / Software QA** | Abstract Syntax Tree (AST) function chunks | Code Embedder (Voyage-Code) + Graph Call Graph | File path & symbol scope validation |

---

## 3. Mathematical Formalization: Code AST & Call Graph Linking

For codebase RAG, passages are structured as call-graph subgraphs $\mathcal{G}_{\text{code}} = (\mathcal{V}_{\text{func}}, \mathcal{E}_{\text{calls}})$:

$$\text{Context}(f) = \text{SourceCode}(f) \cup \bigcup_{g \in \text{Callers}(f)} \text{Signature}(g) \cup \bigcup_{h \in \text{Callees}(f)} \text{Docstring}(h)$$

This eliminates context blindness when a function depends on types or helper methods defined in other files.

---

## 4. Failure Modes & Mitigations

1. **Cross-Domain Embedding Collapse**:
   - *Failure*: A general-domain embedding model maps two completely different legal terms (*"indemnification"* and *"subrogation"*) to nearby vectors because they both appear in contracts.
   - *Mitigation*: Fine-tune domain bi-encoders or utilize domain-specific frontier models.
2. **Missing Temporal Disclaimers in Financial QA**:
   - *Failure*: A model generates an answer regarding company debt from a 2019 report without noting that the company restructured debt in 2023.
   - *Mitigation*: Prepend mandatory document date headers to all retrieved chunks.

---

## 5. SOLID Principles in Domain Adaptations

- **Single Responsibility (SRP)**: `LegalCaseEngine`, `FinancialCaseEngine`, `CodebaseCaseEngine` manage domain-specific rules.
- **Open/Closed (OCP)**: New vertical domains register via `DomainProfileProtocol`.
- **Liskov Substitution (LSP)**: All domain engines execute `async def execute_case_study(query: str) -> CaseStudyReport`.
- **Dependency Inversion (DIP)**: `CaseStudyRunner` depends on `DomainProfileProtocol`.
