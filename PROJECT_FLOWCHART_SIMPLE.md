# DL Legal AI Project: Simple Flowchart

This project builds an AI system for the legal domain using the AILA 2019 dataset.

The dataset contains:

- 2,914 Indian Supreme Court case documents
- 197 legal statutes
- 50 legal situation queries
- Ground-truth relevance judgments for evaluation

## Simple Project Flowchart

```mermaid
flowchart TD
    A["AILA 2019 Legal Dataset"] --> B["Data Preparation"]

    B --> B1["Clean and parse cases"]
    B --> B2["Clean and parse statutes"]
    B --> B3["Parse legal queries"]
    B --> B4["Parse relevance judgments"]

    B1 --> C["Pipeline 1: Legal Search and RAG"]
    B2 --> C
    B3 --> C
    B4 --> C

    C --> C1["BM25 Search Baseline"]
    C --> C2["Doc2Query Expansion"]
    C --> C3["ColBERT-style Reranking"]
    C --> C4["Evaluation using MAP, MRR, Recall, nDCG"]
    C --> C5["RAG Answer Generation with Citations"]
    C --> C6["Search Analytics Report"]

    C1 --> CO["Pipeline 1 Outputs"]
    C2 --> CO
    C3 --> CO
    C4 --> CO
    C5 --> CO
    C6 --> CO

    CO --> CO1["TREC run files"]
    CO --> CO2["Retrieval metrics"]
    CO --> CO3["RAG answers"]
    CO --> CO4["Citation audit"]
    CO --> CO5["HTML analytics report"]

    B --> D["Pipeline 2: Legal Issue Classification"]
    CO --> D

    D --> D1["Create legal label schema"]
    D --> D2["Generate weak labels"]
    D --> D3["Extract legal text features"]
    D --> D4["Train rule-assisted classifier"]
    D --> D5["Predict labels for cases, statutes, and queries"]
    D --> D6["Generate classification report"]

    D1 --> DO["Pipeline 2 Outputs"]
    D2 --> DO
    D3 --> DO
    D4 --> DO
    D5 --> DO
    D6 --> DO

    DO --> DO1["Case labels"]
    DO --> DO2["Statute labels"]
    DO --> DO3["Query labels"]
    DO --> DO4["Classification metrics"]
    DO --> DO5["HTML classification report"]

    B --> E["Pipeline 3: Legal Document Summarization"]
    CO --> E
    DO --> E

    E --> E1["Segment long legal documents"]
    E --> E2["Summarize case judgments"]
    E --> E3["Summarize statutes"]
    E --> E4["Create query-focused summaries"]
    E --> E5["Validate citations and summary quality"]
    E --> E6["Generate summarization report"]

    E1 --> EO["Pipeline 3 Outputs"]
    E2 --> EO
    E3 --> EO
    E4 --> EO
    E5 --> EO
    E6 --> EO

    EO --> EO1["2,914 case summaries"]
    EO --> EO2["197 statute summaries"]
    EO --> EO3["50 query-focused summaries"]
    EO --> EO4["Summary quality report"]
    EO --> EO5["HTML summarization report"]

    CO --> F["Final Pipeline: Model Benchmarking"]
    DO --> F
    EO --> F

    F --> F1["Check ML environment"]
    F --> F2["Run dense retrieval fallback"]
    F --> F3["Run reranking fallback"]
    F --> F4["Prepare transformer classification outputs"]
    F --> F5["Prepare transformer summarization outputs"]
    F --> F6["Prepare model-based RAG outputs"]
    F --> F7["Compare all methods"]

    F1 --> FO["Final Pipeline Outputs"]
    F2 --> FO
    F3 --> FO
    F4 --> FO
    F5 --> FO
    F6 --> FO
    F7 --> FO

    FO --> FO1["Environment report"]
    FO --> FO2["Model benchmark metrics"]
    FO --> FO3["Model install requirements"]
    FO --> FO4["Final comparison report"]

    FO --> Z["Complete Legal AI Suite"]

    Z --> Z1["Search legal cases and statutes"]
    Z --> Z2["Generate citation-based legal answers"]
    Z --> Z3["Classify legal issues"]
    Z --> Z4["Summarize judgments and statutes"]
    Z --> Z5["Benchmark future ML models"]
```

## What We Built In Simple Words

### Pipeline 1: Legal Search and RAG

This pipeline searches relevant prior cases and statutes for a legal query.  
It also generates citation-based legal answers using retrieved documents.

Main outputs:

- TREC retrieval run files
- Evaluation metrics
- RAG answers
- Citation audit
- Analytics report

### Pipeline 2: Legal Issue Classification

This pipeline assigns legal issue labels to cases, statutes, and queries.

Example labels:

- Criminal law
- Constitutional law
- Property law
- Contract law
- Evidence law
- Writ jurisdiction
- Statutory interpretation

Main outputs:

- Case predictions
- Statute predictions
- Query predictions
- Classification metrics
- Classification report

### Pipeline 3: Legal Summarization

This pipeline creates readable summaries of legal documents.

It summarizes:

- 2,914 case documents
- 197 statutes
- 50 query-focused legal situations

Main outputs:

- Case summaries
- Statute summaries
- Query-focused summaries
- Citation validation report
- Summarization analytics report

### Final Pipeline: Model Benchmarking

This pipeline prepares the project for advanced ML and transformer models.

It checks whether libraries like PyTorch, Transformers, Sentence Transformers, FAISS, and ColBERT are installed.  
Because they are not currently installed, it runs fallback versions and saves installation instructions for future upgrades.

Main outputs:

- Environment readiness report
- Model benchmark run files
- Model benchmark metrics
- Final comparison report
- Requirements file for future model installation

## Final Result

The project is now a complete Legal AI suite with:

- Legal information retrieval
- RAG-based legal answer generation
- Legal issue classification
- Legal judgment summarization
- Model benchmarking for future deep learning upgrades

