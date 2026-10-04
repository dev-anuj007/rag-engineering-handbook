# Graph RAG & Knowledge Graph Traversal: Theoretical Deep Dive

## 1. First-Principles Mechanics of Graph RAG

Graph RAG combines unstructured vector similarity with **structured Knowledge Graph (KG) entity-relation topologies** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ to answer complex global questions ("*How are all holding entities in Europe connected to Subsidiary X?*") where standalone vector search fails.

```
                                  [ KNOWLEDGE GRAPH TOPOLOGY ]

[Entity: Alice] ──(CEO_OF)──► [Entity: Acme Corp] ──(ACQUIRED)──► [Entity: Beta AI]
       │                                                                  │
   (AUTHOR_OF)                                                       (LOCATED_IN)
       ▼                                                                  ▼
[Doc: Q3 Report]                                                   [Entity: Dublin]
```

---

## 2. Mathematical Formalization: Graph Traversal & Leiden Community Detection

### 1. Triplet Extraction:
During ingestion, LLMs extract entity-relation-entity triplets from passage $c$:

$$\mathcal{T}_c = \{(e_{\text{head}}, r, e_{\text{tail}}) \mid e \in \mathcal{E}, r \in \mathcal{R}\}$$

### 2. $N$-Hop Subgraph Neighborhood Extraction:
Given entity node $u$ identified in query $q$, retrieve its $N$-hop neighborhood subgraph:

$$\mathcal{N}_N(u) = \{v \in \mathcal{V} \mid \text{dist}_{\mathcal{G}}(u, v) \le N\}$$

### 3. Hierarchical Community Summaries (Microsoft GraphRAG):
Graph RAG applies the **Leiden Community Detection Algorithm** to partition $\mathcal{G}$ into modular clusters at varying hierarchy levels $\{C_1, C_2, \dots, C_L\}$. For each community cluster $C_k$, an LLM generates a comprehensive abstractive summary $\mathcal{S}(C_k)$.
- Global broad queries query the community summaries directly.
- Specific entity queries execute local $N$-hop subgraph traversals.

---

## 3. Vector RAG vs Graph RAG Trade-Off Matrix

| Dimension | Standard Vector RAG | Graph RAG |
|---|---|---|
| **Specific Fact Retrieval** (*"What is Alice's email?"*) | Very Fast ($< 10\text{ ms}$) | Moderate ($\sim 30\text{ ms}$) |
| **Global Thematic Queries** (*"What are the overarching risks across all 50 SEC filings?"*) | Fails (Cannot summarize 1000 chunks) | State-of-the-Art (Community summaries) |
| **Multi-Hop Relational Links** | Weak (Semantic drift across hops) | Deterministic & Exact |
| **Ingestion Compute & Index Cost** | Low ($1\times$) | High ($10\times - 20\times$ due to LLM triplet extraction) |

---

## 4. Failure Modes & Mitigations

1. **Entity Ambiguity & Disambiguation Drift**:
   - *Failure*: Mentions of *"Apple"* (fruit) vs *"Apple Inc."* (company) become merged into a single corrupted node.
   - *Mitigation*: Perform entity resolution using Wikidata / DBpedia entity linking or vector similarity thresholding on entity descriptions.
2. **Exponential Graph Traversal Fan-Out**:
   - *Failure*: In dense graphs, a 3-hop traversal visits 100,000+ nodes, timing out the query.
   - *Mitigation*: Cap neighborhood traversal to $N \le 2$ hops and prune edges using Personalized PageRank (PPR).

---

## 5. SOLID Principles in Graph RAG

- **Single Responsibility (SRP)**: `TripletExtractor` extracts graph triples; `GraphStore` executes traversal queries; `GraphSynthesizer` generates responses.
- **Open/Closed (OCP)**: Graph backends (NetworkX, Neo4j, SQLite Graph) implement `GraphStoreProtocol`.
- **Liskov Substitution (LSP)**: All graph engines return standardized `Subgraph(nodes: list[Node], edges: list[Edge])`.
- **Dependency Inversion (DIP)**: `GraphRAGPipeline` depends on `GraphStoreProtocol` and `VectorStoreProtocol`.
