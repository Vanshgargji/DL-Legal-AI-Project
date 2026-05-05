# AILA 2019 Legal AI Pipeline 1

This project implements Pipeline 1: an offline-first legal IR and RAG engine for the AILA 2019 dataset.

## Run Everything

```powershell
python scripts/run_pipeline1.py --root "D:\DL Legal AI Project" all
```

## Run One Module

```powershell
python scripts/run_pipeline1.py --root "D:\DL Legal AI Project" ingest
python scripts/run_pipeline1.py --root "D:\DL Legal AI Project" bm25
python scripts/run_pipeline1.py --root "D:\DL Legal AI Project" doc2query
python scripts/run_pipeline1.py --root "D:\DL Legal AI Project" colbert
python scripts/run_pipeline1.py --root "D:\DL Legal AI Project" evaluate
python scripts/run_pipeline1.py --root "D:\DL Legal AI Project" rag
python scripts/run_pipeline1.py --root "D:\DL Legal AI Project" analytics
```

## Outputs

- `data/processed/`: parsed documents, queries, qrels, metadata
- `indexes/`: BM25 indexes and ColBERT fallback manifest
- `runs/`: TREC-compatible run files
- `metrics/`: evaluation metrics
- `rag_outputs/`: answers, traces, citation audit
- `analytics/`: comparison CSV, overlap summary, HTML report
- `logs/`: last-run manifests

## Note On ColBERT And Doc2Query

The current environment has no local ML libraries installed, so Doc2Query uses a deterministic legal-keyphrase expansion fallback and ColBERT uses a pure-Python late-interaction fallback. The module boundaries and output files are intentionally stable so a local GPU-backed model can replace either fallback later without changing the rest of the pipeline.

## Pipeline 2: Legal Issue Classification

Run the full classification pipeline:

```powershell
python scripts/run_pipeline2.py --root "D:\DL Legal AI Project" all
```

Pipeline 2 outputs are saved under `classification_outputs/`.

## Pipeline 3: Legal Judgment Summarization

Run the full summarization pipeline:

```powershell
python scripts/run_pipeline3.py --root "D:\DL Legal AI Project" all
```

Pipeline 3 outputs are saved under `summarization_outputs/`.

## Final Pipeline: Model Benchmarking And Upgrade Harness

Run the final model benchmarking pipeline:

```powershell
python scripts/run_model_benchmark.py --root "D:\DL Legal AI Project" all
```

The current implementation records model readiness, runs dependency-free benchmark fallbacks, and saves upgrade instructions under `model_benchmark_outputs/`.
