# RAG System Design & Sizing Architecture: Theoretical Deep Dive

## 1. First-Principles System Sizing & Capacity Planning

Architecting an enterprise RAG cluster requires rigorous mathematical modeling across **storage capacity**, **index RAM/VRAM**, **query throughput (QPS)**, and **p95 latency budget allocation**.

```
                      [ DISTRIBUTED RAG CLUSTER TOPOLOGY ]

                                Client Ingress
                                      │
                                      ▼
                      ┌──────────────────────────────┐
                      │ High-Availability L4/L7 LB   │
                      └──────────────┬───────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌───────────────────────────┐                       ┌───────────────────────────┐
│ Stateless Query Workers   │                       │ Ingestion Workers (Async) │
│ (10x Pods, Autoscale HPA) │                       │ (Queue-driven via Celery) │
└─────────┬─────────────────┘                       └─────────┬─────────────────┘
          │                                                   │
          ▼                                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Distributed Vector Database Cluster (ChromaDB / Persistent Storage Engine)    │
│ Primary Write Master  ◄───►  Horizontal Read Replicas (HNSW Graph Indices)     │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formalization: Back-of-the-Envelope Capacity Math

### 1. Vector Memory Sizing Formula:

$$\text{RAM}_{\text{total}} = N \times \left( D \times 4\text{ bytes} + M \times 2 \times 4\text{ bytes (HNSW pointers)} + \text{Metadata}_{\text{avg}} \right) \times 1.25 \text{ (OS/Buffer)}$$

- *Example*: Sizing for $50\text{ Million chunks}$, $D = 1536$, $M = 32$, $\text{Metadata} = 256\text{ bytes}$:
  - Raw Vector Size: $50\text{M} \times 1536 \times 4 = 307.2\text{ GB}$
  - HNSW Graph Pointers: $50\text{M} \times 32 \times 8 = 12.8\text{ GB}$
  - Metadata: $50\text{M} \times 256 = 12.8\text{ GB}$
  - Total RAM Required: $(307.2 + 12.8 + 12.8) \times 1.25 \approx \mathbf{416\text{ GB RAM}}$.
  - *Cluster Provisioning*: $8\times \text{r6g.2xlarge (64GB RAM each)}$.

### 2. End-to-End Latency Budget Allocation (500ms p95 SLA):

$$\text{Budget} = \underbrace{T_{\text{Gateway}}}_{10\text{ms}} + \underbrace{T_{\text{QueryEmbed}}}_{25\text{ms}} + \underbrace{T_{\text{VectorANN}}}_{15\text{ms}} + \underbrace{T_{\text{CrossEncoder}}}_{40\text{ms}} + \underbrace{T_{\text{PromptPrep}}}_{10\text{ms}} + \underbrace{T_{\text{LLM\_TTFT}}}_{250\text{ms}} + \underbrace{T_{\text{StreamingBuffer}}}_{150\text{ms}} = \mathbf{500\text{ms}}$$

---

## 3. Distributed Sharding & Replication Trade-Off Matrix

| Strategy | Read Scalability | Write Throughput | Failure Recovery Time |
|---|---|---|---|
| **Single Master + $N$ Read Replicas** | High ($N \times \text{QPS}$) | Limited to 1 master | $< 10\text{s}$ (Replica promotion) |
| **Hash-Partitioned Sharding (Consistent Hashing)** | Very High | Very High (Distributed writes) | Moderate (Cross-node rebalancing) |
| **Active-Active Multi-Region Replication** | Global Low Latency | Complex (Requires Conflict Resolution) | Instantaneous DNS failover |

---

## 4. Failure Modes & Mitigations

1. **Hotspot Sharding Degradation**:
   - *Failure*: 80% of query traffic targets a single tenant's data partition, overloading one node while others sit idle.
   - *Mitigation*: Implement consistent hash ring partitioning with virtual nodes and secondary semantic caching.
2. **LLM Provider Concurrency Exhaustion**:
   - *Failure*: Peak traffic (5,000 QPS) exceeds third-party LLM rate limits.
   - *Mitigation*: Provision dedicated throughput (PTU) instances or dynamically load balance across multiple frontier providers.

---

## 5. SOLID Principles in System Design Architecture

- **Single Responsibility (SRP)**: `CapacityCalculator` computes RAM/VRAM math; `ClusterSizer` determines node topology.
- **Open/Closed (OCP)**: Hardware spec models extend `HardwareProfileProtocol`.
- **Liskov Substitution (LSP)**: All sizing estimators implement `SizingEstimatorProtocol`.
- **Dependency Inversion (DIP)**: Capacity planners depend on abstract `CapacityCalculatorProtocol`.
