# 04 Document Parsing & Multimodal Data

## Objective

Extract structured content (prose, tabular rows, and visual diagram representations) from complex documents into strongly-typed LlamaIndex nodes indexed into ChromaDB.

## Why It Matters

Naive text extraction destroys table structures and discards diagrams, creating catastrophic retrieval failures for numerical queries or visual schema questions.

## Core Concepts

- **Modality Segregation**: Discriminating prose, tabular structures, and diagram metadata.
- **Node Typing**: Tagging nodes with modality metadata to enable modality-aware retrieval filters.
- **ChromaDB Unified Vector Indexing**: Storing multimodal node embeddings with indexed metadata tags.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Parsing Mechanics

Document parsing is the lossy translation of layout-heavy, multi-modal documents (PDFs, PPTXs, HTML, spreadsheets) into semantically coherent textual and structured representations.

```
                    ┌────────────────────────────────────────┐
                    │ Multimodal Document (PDF / Scan / PPT) │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Layout Analysis & Object Detection     │
                    │   - Text Blocks (Paragraphs / Headers) │
                    │   - Tabular Grids (Row / Col Spans)    │
                    │   - Embedded Images & Figures          │
                    └───────────────────┬────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐               ┌───────────────┐               ┌───────────────┐
│ Text Extractor│               │ Table Parser  │               │ Vision OCR    │
│ (PyPDF/MuPDF) │               │ (Markdown/HTML│               │ (VLM / Tesseract
└───────┬───────┘               └───────┬───────┘               └───────┬───────┘
        │                               │                               │
        │ Text Chunks                   │ Markdown Tables               │ Image Captions
        └───────────────────────────────┼───────────────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Unified Structured Markdown Stream     │
                    │ Header hierarchy preserved (#, ##, ###)│
                    │ Tables preserved as GFM Markdown tables│
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Ingestion / Chunking Pipeline          │
                    └────────────────────────────────────────┘
```

---

## 2. Theoretical Mechanics: Tabular & Multimodal Preservation

### Table Structure Serialization
Naive text extraction flattens multi-column PDF tables into an unreadable sequence of numbers and labels, destroying row-column associations.
To preserve relational context:
1. **Geometric Grid Alignment**: Detect bounding boxes for cell coordinates $(x_0, y_0, x_1, y_1)$.
2. **Markdown / HTML Serialization**: Convert tabular structures into GitHub-Flavored Markdown (GFM) tables:

$$\text{Table Serialization} = \text{Concat}(\text{Headers}) + \sum_{r \in \text{Rows}} \left(\text{RowPrefix} \parallel \text{Cells}(r) \right)$$

This guarantees that dense bi-encoders and LLMs retain relational attribute bindings (e.g., "Revenue in Q3 2024").

### Multimodal Vision-Language Models (VLM) for Complex Charts
When extracting figures, diagrams, or scanned handwriting:
- Extract bounding box $\mathcal{B}_{\text{image}}$.
- Pass image tensor to a fast VLM (e.g. Gemini 1.5 Flash) with an extraction prompt: *"Transcribe all data points, trend lines, and axes into dense descriptive markdown."*
- Inject generated description directly into the document stream wrapped in `[FIGURE: ...]`.

---

## 3. Parsing Strategy Trade-Off Matrix

| Parsing Method | Ingestion Cost | Speed (Pages/Sec) | Layout & Table Fidelity | Scanned OCR Support |
|---|---|---|---|---|
| **Simple PyPDF / PDFMiner** | $\$0.00$ | Very Fast ($50\text{ pps}$) | Very Low (Flattens columns) | No (Returns empty text) |
| **Rule-Based Layout (Unstructured/MuPDF)** | Low | Fast ($15\text{ pps}$) | Medium (Basic table grids) | Moderate (via Tesseract) |
| **Vision Model / OCR API (LlamaParse/Gemini)** | $\$0.002 - \$0.01\text{/page}$ | Moderate ($2\text{ pps}$) | Very High (Pixel-perfect tables) | Flawless |

---

## 4. Failure Modes & Mitigations

1. **Multi-Column Reading Order Scrambling**:
   - *Failure*: Parser reads horizontally across a two-column academic paper, interleaving column 1 line 1 with column 2 line 1.
   - *Mitigation*: Perform connected-component layout analysis to segment column boundaries before reading text tokens.
2. **Token Explosion in Dense HTML / XML Tables**:
   - *Failure*: Verbose HTML tags consume 70% of the chunk token limit without adding semantic value.
   - *Mitigation*: Strip styling attributes (`style`, `class`, `id`) and convert table to minimal compact Markdown representation.

---

## 5. SOLID Principles in Document Parsing

- **Single Responsibility (SRP)**: `PdfParser` handles PDF decoding; `TableExtractor` reconstructs tables; `ImageCaptioner` interfaces with VLMs.
- **Open/Closed (OCP)**: New file format handlers (e.g. `.docx`, `.epub`, `.pptx`) register with `ParserRegistry` without changing pipeline logic.
- **Liskov Substitution (LSP)**: All parsers implement `DocumentParserProtocol` returning standardized `ParsedDocument` AST objects.
- **Dependency Inversion (DIP)**: Extraction orchestrator depends on abstract `DocumentParserProtocol`.

---

## Implementation

Implemented in [multimodal_parser.py](implementation/multimodal_parser.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Preserves tabular context and visual semantic cues.
- **Weaknesses**: Requires more complex preprocessing pipelines.

## Architectural Decision & Trade-off Questions

1. *How do you represent complex multi-page financial tables in a RAG system without losing column-header associations across chunk boundaries?*
2. *When should you embed image summary text vs using a joint multimodal embedding space (e.g. CLIP / Gemini Multimodal Embeddings)?*
