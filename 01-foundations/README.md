# RAG Foundations

## 1. What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that combines:

1. **Retrieval** — finding relevant information from an external knowledge source.
2. **Augmentation** — providing that information as context to a language model.
3. **Generation** — using the language model to produce an answer grounded in that context.

The basic idea is:

```text
User Query
    │
    ▼
Retrieve relevant information
    │
    ▼
Add retrieved information to the prompt
    │
    ▼
LLM
    │
    ▼
Grounded Answer
```

A traditional LLM generates an answer primarily from knowledge encoded in its parameters.

A RAG system instead gives the model access to **external, query-time information**.

This external information can come from:

* Internal company documents
* PDFs
* Databases
* Knowledge bases
* Websites
* Research papers
* Product documentation
* Customer records
* Code repositories
* APIs
* Enterprise data stores

---

# 2. The Problem RAG Solves

LLMs are powerful reasoning and generation systems, but their parameters are not a reliable real-time database of an organization's knowledge.

Consider an enterprise application with:

```text
10,000 PDF documents
5 million database records
Internal policies
Customer information
Product documentation
Research reports
Security reports
```

The LLM was not trained on this private information.

Even when information appeared in training data, it may be:

* outdated
* incomplete
* inaccessible
* ambiguous
* difficult to trace to a source

RAG provides a mechanism to retrieve the required information at query time.

---

# 3. Why LLMs Alone Are Not Enough

## 3.1 Knowledge Cutoff

The model may not know information created after its training data.

```text
Training Data
      │
      ▼
   LLM Model
      │
      ▼
Knowledge available to model
```

RAG adds:

```text
Current / Private Knowledge
          │
          ▼
       Retrieval
          │
          ▼
      LLM Context
```

---

## 3.2 Private Enterprise Knowledge

Suppose an employee asks:

> "What is our company's reimbursement policy for international travel?"

The answer may exist in an internal HR document.

The LLM itself does not automatically have access to that document.

RAG can retrieve it:

```text
Question
   │
   ▼
HR Knowledge Base
   │
   ▼
Travel Policy.pdf
   │
   ▼
Relevant section
   │
   ▼
LLM
```

---

## 3.3 Freshness

Enterprise information changes.

Examples:

* Product pricing
* Security policies
* Financial reports
* Legal documents
* Product specifications
* Employee policies

RAG allows the knowledge layer to be updated independently from the LLM.

This is one of the most important architectural properties of RAG.

---

## 3.4 Traceability

A well-designed RAG system can return:

```text
Answer
   +
Source document
   +
Page
   +
Section
   +
Retrieved chunk
```

This allows users and downstream systems to inspect the evidence behind an answer.

---

## 3.5 Domain-Specific Knowledge

A general-purpose LLM may know general financial concepts but not a company's:

* internal financial terminology
* research methodology
* proprietary models
* internal processes
* customer-specific information

RAG allows domain knowledge to be injected at query time.

---

# 4. RAG Does Not "Teach" the LLM

This distinction is fundamental.

RAG does **not** modify the model's weights.

```text
                 ┌───────────────┐
                 │   LLM Model   │
                 │               │
                 │  Parameters   │
                 └───────────────┘
                         ▲
                         │
                    Context
                         │
                         │
                 ┌───────┴───────┐
                 │   Retrieval   │
                 └───────┬───────┘
                         ▲
                         │
                    Knowledge
```

The model remains unchanged.

The retrieved information is provided as context during inference.

Compare this with fine-tuning:

```text
Fine-tuning

Dataset
   │
   ▼
Training
   │
   ▼
Modified Model Weights
```

RAG:

```text
Documents
   │
   ▼
Index
   │
   ▼
Retrieval
   │
   ▼
Context
   │
   ▼
Existing LLM
```

This distinction will become important when comparing RAG with fine-tuning later.

---

# 5. Basic RAG Architecture

A basic RAG system has two major pipelines.

## 5.1 Offline / Indexing Pipeline

This pipeline prepares knowledge before users ask questions.

```text
Documents
    │
    ▼
Document Parsing
    │
    ▼
Chunking
    │
    ▼
Metadata Extraction
    │
    ▼
Embedding Generation
    │
    ▼
Vector Index
```

For example:

```text
company_policy.pdf
        │
        ▼
    Parse PDF
        │
        ▼
    100 chunks
        │
        ▼
Generate embeddings
        │
        ▼
Vector Database
```

This is generally called the **ingestion/indexing pipeline**.

---

# 6. Online / Query Pipeline

This pipeline executes when the user asks a question.

```text
User Query
    │
    ▼
Query Processing
    │
    ▼
Retrieval
    │
    ▼
Candidate Chunks
    │
    ▼
Reranking
    │
    ▼
Context Construction
    │
    ▼
LLM
    │
    ▼
Answer
```

For example:

```text
"What is our international travel reimbursement limit?"
                     │
                     ▼
               Query Embedding
                     │
                     ▼
              Vector Retrieval
                     │
                     ▼
           Top 10 candidate chunks
                     │
                     ▼
                 Reranker
                     │
                     ▼
             Top 3 relevant chunks
                     │
                     ▼
                    LLM
                     │
                     ▼
        "The limit is ₹X per day..."
```

---

# 7. RAG Request Lifecycle

A complete RAG request can be viewed as a sequence of decisions.

```text
                    USER QUERY
                        │
                        ▼
                ┌───────────────┐
                │ Query Analysis │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │   Retrieval   │
                └───────┬───────┘
                        │
                        ▼
              Candidate Documents
                        │
                        ▼
                ┌───────────────┐
                │   Reranking   │
                └───────┬───────┘
                        │
                        ▼
                 Relevant Context
                        │
                        ▼
                ┌───────────────┐
                │     Prompt    │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │      LLM      │
                └───────┬───────┘
                        │
                        ▼
                     Answer
```

Every stage can independently fail.

Therefore, RAG quality cannot be reduced to simply asking whether the final answer is correct.

We need to understand **where the failure occurred**.

---

# 8. Core RAG Components

## 8.1 Document

A document is the original knowledge source.

Examples:

```text
PDF
DOCX
HTML
Markdown
CSV
JSON
Database record
Web page
Email
```

---

## 8.2 Node / Chunk

Large documents are usually divided into smaller units.

Example:

```text
100-page PDF
      │
      ▼
  500 chunks
```

A chunk should contain enough information to be useful for retrieval without becoming unnecessarily large.

Chunking is covered deeply in:

* Chapter 05 — Chunking Fundamentals
* Chapter 06 — Advanced Chunking

---

## 8.3 Embedding

An embedding converts information into a numerical vector.

Conceptually:

```text
"Python is a programming language"
                 │
                 ▼
        Embedding Model
                 │
                 ▼
[0.12, -0.43, 0.87, ...]
```

Embeddings allow the system to compare the semantic relationship between queries and knowledge.

Embeddings receive dedicated treatment in:

* Chapter 07 — Embedding Fundamentals
* Chapter 08 — Embedding Models
* Chapter 09 — Embedding Selection

---

## 8.4 Vector Store

The generated embeddings need to be stored somewhere.

A vector store typically contains:

```text
Vector
+
Original/derived text
+
Metadata
+
Document identifiers
```

Example:

```python
{
    "id": "chunk_123",
    "vector": [...],
    "text": "...",
    "metadata": {
        "document": "policy.pdf",
        "page": 15,
        "department": "HR"
    }
}
```

Examples of vector-capable databases include:

* Pinecone
* Qdrant
* Weaviate
* Milvus
* pgvector
* OpenSearch
* Elasticsearch

Vector databases are covered in Chapter 10.

---

## 8.5 Retriever

The retriever receives a query and searches the knowledge index.

```text
Query
  │
  ▼
Retriever
  │
  ├── Chunk A
  ├── Chunk B
  ├── Chunk C
  └── Chunk D
```

The retriever's job is:

> Find potentially relevant information.

It does **not** necessarily determine the final answer.

Retrieval is covered in Chapters 12 and 13.

---

## 8.6 Reranker

The initial retriever may return:

```text
Top 20 candidates
```

A reranker can examine the query and candidates more deeply:

```text
20 candidates
      │
      ▼
   Reranker
      │
      ▼
Top 5 relevant chunks
```

This creates a common two-stage retrieval architecture:

```text
Stage 1
Fast candidate retrieval
        ↓
Stage 2
More expensive relevance ranking
```

Reranking is covered in Chapter 14.

---

## 8.7 Context

The selected chunks become context for the LLM.

```text
User Query
+
Retrieved Context
        │
        ▼
      Prompt
```

Example:

```text
SYSTEM:
Answer using only the provided context.

CONTEXT:
[Document: travel_policy.pdf]
Employees can claim ...

QUESTION:
What is the international travel limit?
```

Context engineering is covered in Chapter 15.

---

## 8.8 Generator

The LLM uses:

```text
Query
+
Instructions
+
Retrieved Context
```

to generate the final response.

The generator can be:

* GPT
* Claude
* Gemini
* Llama
* Mistral
* Qwen
* other compatible language models

Generation and synthesis are covered in Chapter 16.

---

# 9. RAG vs LLM-Only

| Aspect                  | LLM Only                          | RAG                           |
| ----------------------- | --------------------------------- | ----------------------------- |
| Private knowledge       | Limited                           | Yes                           |
| Fresh information       | Limited                           | Yes                           |
| External knowledge      | No direct access                  | Yes                           |
| Citations               | Difficult                         | Possible                      |
| Knowledge updates       | Requires model update/fine-tuning | Update knowledge index        |
| Retrieval latency       | Lower                             | Higher                        |
| Architecture complexity | Low                               | Higher                        |
| Hallucination           | Possible                          | Can be reduced with grounding |
| Source traceability     | Limited                           | Stronger                      |

RAG does **not automatically eliminate hallucinations**.

Poor retrieval can actually provide misleading context.

---

# 10. RAG vs Fine-Tuning

These solve different problems.

### RAG

Best suited for providing **external or changing knowledge**.

```text
Knowledge
    ↓
Retrieval
    ↓
Context
    ↓
LLM
```

### Fine-tuning

Primarily changes model behavior through training.

```text
Training Data
     ↓
Fine-tuning
     ↓
Modified Weights
```

A useful engineering principle is:

> **Use RAG to provide knowledge at inference time; use fine-tuning when you need to change model behavior.**

The two approaches can also be combined.

---

# 11. RAG vs Long Context

Modern LLMs can process very large contexts.

This raises an important question:

> If the model can accept millions of tokens, why do we need retrieval?

Because putting everything into context can create:

* higher cost
* higher latency
* irrelevant information
* context competition
* reduced retrieval precision
* difficulty identifying the most relevant evidence

Instead of:

```text
1,000,000 tokens
       ↓
      LLM
```

RAG attempts to provide:

```text
1,000,000 tokens
       ↓
   Retrieval
       ↓
10,000 relevant tokens
       ↓
      LLM
```

Long-context models and RAG are not mutually exclusive.

A production system can use both.

---

# 12. RAG vs Tools

A tool generally performs an **action or query against a system**.

Examples:

```text
get_customer()
execute_sql()
get_stock_price()
create_ticket()
send_email()
```

RAG primarily retrieves knowledge.

For example:

```text
RAG:
"What does our refund policy say?"

Tool:
"Create a refund request for customer 123."
```

A production agent may use both.

---

# 13. RAG vs Agents

RAG is primarily a knowledge retrieval architecture.

An agent introduces decision-making and dynamic execution.

### Traditional RAG

```text
Query
 ↓
Retrieve
 ↓
Generate
```

### Agentic system

```text
User
 ↓
Agent
 ↓
Decide what to do
 ↓
Tool / RAG / API / Database
 ↓
Observe result
 ↓
Reason
 ↓
Another action
 ↓
Final answer
```

Agentic RAG is covered separately in Chapter 18.

---

# 14. Types of RAG

RAG has evolved into several architectural patterns.

## 14.1 Naive RAG

The simplest architecture:

```text
Query
 ↓
Vector Search
 ↓
Top-K chunks
 ↓
LLM
```

Useful for prototypes and simple knowledge bases.

---

## 14.2 Advanced RAG

Adds techniques such as:

* better chunking
* metadata filtering
* query rewriting
* hybrid retrieval
* reranking
* context compression

---

## 14.3 Modular RAG

Each component can be independently replaced.

```text
Parser
   ↓
Chunker
   ↓
Embedding
   ↓
Retriever
   ↓
Reranker
   ↓
Generator
```

This makes experimentation and optimization easier.

---

## 14.4 Hybrid RAG

Combines multiple retrieval mechanisms.

Typical example:

```text
             Query
               │
        ┌──────┴──────┐
        ▼             ▼
   Vector Search    BM25
        │             │
        └──────┬──────┘
               ▼
            Fusion
               │
               ▼
            Reranker
               │
               ▼
              LLM
```

---

## 14.5 Graph RAG

Graph RAG represents entities and relationships.

Example:

```text
Company A
    │
    ├── owns ──► Company B
    │
    └── uses ──► Product X
                     │
                     └── affected by ──► CVE-1234
```

It is particularly useful for:

* multi-hop reasoning
* relationship-heavy questions
* organizational knowledge
* entity-centric queries

Graph RAG is covered deeply in Chapter 30.

---

## 14.6 Multimodal RAG

Retrieves multiple information modalities:

```text
Text
Images
Tables
Charts
Audio
Video
```

Example:

```text
Question
   ↓
Retrieve:
   ├── Text chunk
   ├── Financial table
   └── Chart
   ↓
Multimodal LLM
```

---

## 14.7 Agentic RAG

The system dynamically decides how retrieval should happen.

```text
Question
   ↓
Agent
   ↓
Should I retrieve?
   │
   ├── Search documents
   ├── Search database
   ├── Rewrite query
   ├── Retrieve again
   └── Verify evidence
```

---

# 15. When Should We Use RAG?

RAG is a strong candidate when:

### Knowledge changes frequently

```text
Policies
Pricing
Documentation
Research
Security information
```

### Knowledge is private

```text
Internal documents
Customer data
Company knowledge
```

### Source attribution matters

```text
Research
Legal
Financial
Enterprise decision support
```

### The knowledge base is large

It is impractical to place the entire corpus into every prompt.

### You need independent knowledge updates

The knowledge layer can change without retraining the LLM.

---

# 16. When Should We NOT Use RAG?

RAG adds architectural complexity.

It may not be necessary when:

### Simple conversational task

```text
"Write a professional email."
```

No external knowledge is required.

### Pure transformation

```text
Translate this text.
Summarize this provided paragraph.
Convert JSON to YAML.
```

The required information is already in the prompt.

### Deterministic computation

A calculator or programmatic operation may be more appropriate.

### The source of truth is an API

For example:

```text
"What is my current account balance?"
```

A live banking API is generally preferable to retrieving an old document containing a balance.

---

# 17. Simple RAG Example

Suppose we have:

```text
company_policy.pdf
```

The document contains:

```text
Employees can claim up to ₹5,000 per day
for international meals.
```

The user asks:

> What is the international meal reimbursement limit?

The pipeline becomes:

```text
                 company_policy.pdf
                         │
                         ▼
                      Parsing
                         │
                         ▼
                      Chunking
                         │
                         ▼
                    Embedding
                         │
                         ▼
                   Vector Store
                         │
                         │
User Query ──────────────┘
                         │
                         ▼
                      Retrieval
                         │
                         ▼
                 Relevant chunk
                         │
                         ▼
                       LLM
                         │
                         ▼
          "₹5,000 per day."
```

The LLM did not need to memorize the policy.

It received the relevant information at inference time.

---

# 18. LlamaIndex Mental Model

LlamaIndex provides abstractions for building RAG pipelines.

A simplified mapping is:

| RAG Concept         | LlamaIndex Concept         |
| ------------------- | -------------------------- |
| Document            | `Document`                 |
| Chunk               | `Node`                     |
| Chunking            | `NodeParser`               |
| Index               | `VectorStoreIndex`         |
| Retrieval           | `Retriever`                |
| Query orchestration | `QueryEngine`              |
| Reranking/filtering | `NodePostprocessor`        |
| Generation          | LLM / Response Synthesizer |
| Evaluation          | Evaluators                 |
| Complex workflows   | Workflows                  |

The important principle is:

> **Understand the RAG architecture first; use LlamaIndex as the implementation framework.**

The handbook will use LlamaIndex extensively, but the underlying concepts should remain framework-independent.

---

# 19. RAG Quality Boundaries

One of the most important concepts in this handbook is that a RAG system has multiple independent quality boundaries.

```text
Document
   │
   ▼
Parsing Quality
   │
   ▼
Chunking Quality
   │
   ▼
Embedding Quality
   │
   ▼
Retrieval Quality
   │
   ▼
Reranking Quality
   │
   ▼
Context Quality
   │
   ▼
Generation Quality
   │
   ▼
Final Answer
```

A bad answer does not necessarily mean the LLM failed.

For example:

```text
User Question
      │
      ▼
Retriever
      │
      X
Wrong chunk retrieved
      │
      ▼
LLM
      │
      ▼
Incorrect answer
```

The LLM may have generated a perfectly reasonable answer **from the wrong evidence**.

Therefore:

> **RAG evaluation must evaluate the pipeline, not only the final answer.**

This principle will guide the evaluation chapters.

---

# 20. Strengths

RAG provides several architectural advantages:

* External knowledge can be updated independently.
* Private knowledge can be accessed at inference time.
* Source attribution can be implemented.
* Retrieval can be optimized independently of generation.
* Different embedding models can be experimented with.
* Different vector databases can be substituted.
* Access control can be implemented at retrieval time.
* Retrieval quality can be measured independently.
* The same LLM can serve multiple knowledge domains.
* Knowledge can be added without retraining the generator.

---

# 21. Weaknesses

RAG introduces its own problems:

* Retrieval can return irrelevant information.
* Important information can be missed.
* Poor chunking can destroy context.
* Embedding models may not capture domain semantics.
* Large context can increase cost.
* Reranking increases latency.
* Indexes require maintenance.
* Documents can become stale.
* Access-control mistakes can expose sensitive data.
* Retrieved documents can contain prompt injection attacks.
* Evaluation is significantly more complex than evaluating an LLM-only system.

---

# 22. Fundamental Trade-offs

RAG system design is largely about managing trade-offs.

| Decision                  | Trade-off                                            |
| ------------------------- | ---------------------------------------------------- |
| Small chunks              | Better precision / less context                      |
| Large chunks              | More context / lower precision                       |
| Higher Top-K              | Higher recall / more noise                           |
| Lower Top-K               | Less noise / lower recall                            |
| Better reranker           | Better ranking / higher latency                      |
| Larger embedding model    | Potentially better quality / higher cost             |
| Hybrid retrieval          | Better lexical + semantic coverage / more complexity |
| More context              | Potentially more evidence / higher cost              |
| More aggressive filtering | Less noise / possible missed evidence                |
| Fresh re-indexing         | Fresher knowledge / higher ingestion cost            |

These trade-offs will be explored experimentally throughout the handbook.

---

# 23. Production Considerations

A production RAG system should answer at least these questions.

### Knowledge lifecycle

* Where does the data come from?
* How frequently does it change?
* How do we detect changes?
* How do we handle document updates?
* How do we delete stale data?
* How do we re-index after changing the embedding model?

### Retrieval

* How do we measure retrieval quality?
* How do we handle zero-result queries?
* How do we filter by tenant/user permissions?
* When do we rerank?
* How do we handle conflicting documents?

### Generation

* How do we prevent unsupported claims?
* How do we provide citations?
* When should the system abstain?
* How do we detect hallucinations?

### Operations

* How much does each query cost?
* What is the P50/P95/P99 latency?
* How do we trace retrieval failures?
* How do we handle model/API failures?
* How do we scale ingestion?
* How do we re-index millions of documents?

### Security

* Can one tenant retrieve another tenant's data?
* Can a malicious document inject instructions into the LLM?
* Can retrieved content leak sensitive information?
* Can users manipulate retrieval filters?

These questions become the foundation for the production chapters.

---

# 24. Failure Modes

A RAG system can fail at several layers.

```text
                 RAG FAILURE
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
   Ingestion       Retrieval      Generation
       │              │              │
   Bad parsing     Wrong chunks    Hallucination
   Missing data    Low recall      Ignored context
   Stale data      Bad ranking     Wrong synthesis
```

Common failures include:

### Ingestion failure

The required document was never indexed.

### Parsing failure

The important information was not extracted correctly.

### Chunking failure

A relevant fact was split across chunks.

### Embedding failure

Semantically related content is not represented closely enough.

### Retrieval failure

The correct chunk exists but is not retrieved.

### Ranking failure

The correct chunk is retrieved but ranked too low.

### Context failure

The relevant information is retrieved but not properly included in the prompt.

### Generation failure

The LLM receives the correct evidence but produces an unsupported answer.

This classification will later become the basis for a **systematic RAG debugging methodology**.

---

# 25. Mastery Framework

The goal of this repository is not to memorize RAG terminology.

For every component, we will answer five questions:

```text
                ┌─────────────────────┐
                │   RAG COMPONENT     │
                └──────────┬──────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           WHY?          HOW?       WHEN?
              │            │            │
              ▼            ▼            ▼
           Problem     Implementation  Selection
              │            │            │
              └────────────┼────────────┘
                           ▼
                     HOW GOOD IS IT?
                           │
                           ▼
                       Evaluation
                           │
                           ▼
                    Production Reality
```

For every major RAG technique we will study:

1. **Why does it exist?**
2. **What problem does it solve?**
3. **How does it work internally?**
4. **How do we implement it with LlamaIndex?**
5. **When should we use it?**
6. **When should we avoid it?**
7. **What are its strengths and weaknesses?**
8. **How do we evaluate it?**
9. **What are its failure modes?**
10. **What changes in production?**

This framework will be applied to:

* Parsing
* Chunking
* Embeddings
* Vector databases
* Retrieval
* Hybrid search
* Reranking
* Query transformation
* Context engineering
* Generation
* Graph RAG
* Agentic RAG
* Evaluation
* Security
* Observability

---

# 26. Key Takeaways

1. **RAG is a retrieval + context + generation architecture.**
2. RAG does not modify LLM weights.
3. RAG provides external knowledge at inference time.
4. RAG is especially useful for private, dynamic, and large knowledge bases.
5. RAG is not automatically better than fine-tuning or long context.
6. Retrieval quality and generation quality are separate problems.
7. Chunking, embeddings, retrieval, reranking and context construction all affect final quality.
8. A bad answer does not necessarily mean the LLM failed.
9. Production RAG requires evaluation, security, observability and data lifecycle management.
10. The goal is not simply to retrieve documents.
11. **The goal is to retrieve the right evidence and produce a grounded answer.**
12. Mastering RAG requires understanding the entire pipeline and the trade-offs between its components.

---

# 27. Chapter Summary

The simplest mental model to carry forward is:

```text
                  RAG
                   │
        ┌──────────┴──────────┐
        │                     │
    OFFLINE                  ONLINE
        │                     │
        ▼                     ▼
   Build Knowledge        Answer Query
        │                     │
   ┌────┴────┐          ┌─────┴─────┐
   │         │          │           │
 Parse    Chunk       Retrieve    Generate
   │         │          │           │
   └────┬────┘          └─────┬─────┘
        │                     │
    Embed + Index         Context + LLM
        │                     │
        └─────────┬───────────┘
                  ▼
              Grounded
               Answer
```

Everything else in this handbook is an attempt to make one or more of these stages **more accurate, reliable, secure, observable, scalable, and cost-effective**.

---

## Architectural Decision & Trade-off Questions

1. *Under what conditions would you architect an enterprise knowledge retrieval system using Fine-Tuning or Long-Context Windows instead of a RAG pipeline?*
2. *How do you isolate and attribute failures between the retrieval subsystem and the generation synthesis LLM in an SLA-governed production environment?*
