# Metadata Filtering & Access Control: Theoretical Deep Dive

## 1. First-Principles Mechanics of Secure Multi-Tenancy

In enterprise RAG, search queries must strictly obey Role-Based Access Control (RBAC) and Tenant Isolation. Metadata filtering guarantees that unprivileged users cannot retrieve confidential passages, regardless of semantic similarity.

```
                      [ METADATA FILTERING PARADIGMS ]

1. PRE-FILTERING (Hard Isolation - Mandatory for Security):
   Query + Security Context ──► [Filter by Tenant & Role] ──► Sub-Index ANN Search ──► Results
   (Only search within documents the user is authorized to read)

2. POST-FILTERING (Dangerous for Recall):
   Query ──► [Global ANN Top-100] ──► [Discard Unauthorized Docs] ──► 0-2 Results Left!
   (Severe recall drop when top semantic matches are unauthorized)
```

---

## 2. Mathematical Formalization: Filtered Approximate Nearest Neighbor

Let $\mathcal{D}$ be the global knowledge corpus, $q$ be the query vector, and $\mathcal{F}_{\text{user}}: \mathcal{D} \rightarrow \{0, 1\}$ be the security predicate determined by user role $R_u$ and tenant $T_u$:

$$\mathcal{F}_{\text{user}}(d) = \mathbb{I}\left(\text{tenant}(d) == T_u \land \text{role}(d) \subseteq R_u\right)$$

The filtered retrieval problem is formulated as:

$$\text{Top-}K(\mathcal{D}, q, \mathcal{F}_{\text{user}}) = \arg\max_{\substack{\mathcal{S} \subseteq \mathcal{D}, |\mathcal{S}| = K \\ \forall d \in \mathcal{S}, \mathcal{F}_{\text{user}}(d) = 1}} \sum_{d \in \mathcal{S}} \text{CosineSim}(q, d)$$

### Filter-First Single-Stage HNSW Search:
During HNSW graph traversal, when evaluating candidate neighbors $v \in \mathcal{N}(u)$, the algorithm applies bitset masking:

$$\text{If } \mathcal{F}_{\text{user}}(v) == 0, \text{ skip distance computation and traverse next edge.}$$

---

## 3. Multi-Tenancy Architecture Trade-Offs

| Multi-Tenancy Pattern | Isolation Guarantee | Infrastructure Cost | Index Maintenance Overhead |
|---|---|---|---|
| **Database per Tenant** | Physical / Absolute | High (Separate instances) | Very High ($N$ DB clusters) |
| **Collection per Tenant (ChromaDB)** | Logical / High | Moderate | Low (Separate collection directories) |
| **Shared Collection + Metadata Predicates** | Logical (Requires strict enforcement) | Lowest (Single cluster) | Lowest |

---

## 4. Failure Modes & Mitigations

1. **Post-Filtering Empty Result Sets**:
   - *Failure*: An engineer queries a common term. The top-10 global semantic matches belong to executive HR records. Post-filtering discards all 10, returning 0 results to the user.
   - *Mitigation*: Never use post-filtering for access control. Always apply pre-filtering or single-stage filtered HNSW search in ChromaDB (`where={"tenant_id": "T1", "role": {"$in": ["engineering"]}}`).
2. **Metadata Injection via Malicious Chunk Ingestion**:
   - *Failure*: An attacker ingests a document with forged metadata `{"role": "public"}` to bypass department access gates.
   - *Mitigation*: Sign metadata payloads cryptographically at the ingestion gateway using server-side JWT / HMAC tokens.

---

## 5. SOLID Principles in Access Control

- **Single Responsibility (SRP)**: `SecurityContextEnforcer` parses user roles; `MetadataFilterBuilder` formats database predicates.
- **Open/Closed (OCP)**: New access policy engines (ABAC, RBAC, OAuth scopes) plug into `AccessPolicyProtocol`.
- **Liskov Substitution (LSP)**: Filter builders yield standardized dictionary predicates compatible with `VectorStoreProtocol`.
- **Dependency Inversion (DIP)**: Query pipeline injects `SecurityContextProtocol` into the retriever.
