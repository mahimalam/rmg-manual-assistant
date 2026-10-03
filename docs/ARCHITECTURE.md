# Current development architecture

## Industry purpose and deployment

The research concerns Bengali access to industrial engineering manuals, with garment-industry sewing-machine manuals as the current development domain. The present implementation is a single local Streamlit workspace. Local deployment is a development arrangement; factory rollout and multi-user access are not implemented claims.

## Ingestion and storage

PDFs enter through the manual library. Local extraction combines text/layout processing and English OCR. Structured knowledge separates paragraphs, procedures and tables, retaining source pages and linked notes, prerequisites and subclass exceptions. Optional visual enrichment sends selected pages to a configured vision-capable gateway and caches the result.

SQLite records manuals, knowledge releases, queries, feedback and development trials. PDF files, embeddings, indexes and caches live in ignored runtime storage. Manual replacement and removal update the associated knowledge lifecycle.

## Retrieval and answers

Dense BGE-M3 retrieval and independent BM25 search are combined with reciprocal rank fusion. Optional cross-encoder reranking and dependency expansion select source context. The configured gateway generates answers from retrieved passages. The interface exposes source pages and checks citation identifiers.

Source-derived diagnostic rules check selected settings, conditions and branch polarity. Their current coverage is bounded; they do not establish a physical cause or validate every generated sentence.

## Voice

Streamlit records input audio. Language identification and explicit language choices route recognition to optional Deepgram or local recognition. Prepared question text remains editable. Bengali speech output uses a local model. Recognition metadata and query provenance support development analysis; public source excludes recordings and runtime records.

## Evaluation boundary

Software tests cover contracts, parsing, retrieval dependencies, rejection paths and persistence. Independent domain grading, speech comprehension evaluation, calibrated acceptance criteria and participant qualification remain separate research work. See [TESTING.md](TESTING.md).
