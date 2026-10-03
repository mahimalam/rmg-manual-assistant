# RMG Manual Assistant

**Industry-focused thesis research · Working prototype under active testing**

A Bengali voice-enabled AI assistant for accessing industrial engineering manuals. The research focuses on helping industry teams find and understand information in English machine manuals through Bengali questions and source-referenced answers. Sewing-machine manuals from the ready-made garment (RMG) industry are the current development domain.

The application runs locally during development. This describes its current deployment, not a personal-use purpose. Factory deployment, independent engineering validation and the participant study are still pending.

## What is implemented

- Persistent PDF upload, replacement and removal, with local text/layout extraction and English OCR.
- Structured paragraphs, procedures and table rows with linked notes, prerequisites and model exceptions.
- Hybrid retrieval using BGE-M3 dense embeddings, BM25, reciprocal rank fusion and optional reranking.
- Bengali and English typed questions, voice input, optional Deepgram recognition with local fallback, and local Bengali speech output.
- Answers linked to manual pages, citation checks and source-derived checks for supported diagnostic branches.
- Optional cached visual extraction for difficult pages, with explicit spending controls.
- Testing feedback, query provenance and a development study workspace.

These are implemented capabilities, not claims of independently measured accuracy or industry adoption. The assistant helps users inspect information; it does not autonomously repair or control machinery.

## Run locally

Use Python 3.11 or 3.12. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Tesseract with English language data first.

```sh
uv sync --locked --extra test
mkdir -p ~/.config/rmg-manual-assistant
chmod 700 ~/.config/rmg-manual-assistant
cp .env.example ~/.config/rmg-manual-assistant/.env
chmod 600 ~/.config/rmg-manual-assistant/.env
```

Set `MWAPI_API_KEY` in that private configuration file. `MWAPI_BASE_URL` must be an HTTPS OpenAI-compatible gateway; `MWAPI_MODEL` selects its model. The default configuration reflects the gateway used during development. `DEEPGRAM_API_KEY` is optional.

```sh
./start.sh
```

Open the local address printed by Streamlit, normally `http://127.0.0.1:8501`. Upload a manual in **Manual library**, build its knowledge/index, then ask a question. The public repository starts with an empty library. First use downloads the required embedding and speech models. The lockfile selects CPU PyTorch wheels. Test your microphone and engineering terminology on your target hardware.

PDFs and runtime knowledge remain local. Configured generation requests send retrieved passages to the gateway. Visual extraction sends selected page images; optional Deepgram recognition sends audio. Local recognition is available without Deepgram. Uploading a manual builds an index; it does not retrain a model.

## Verify

```sh
uv run pytest -q
uv run python -m compileall -q app tests evaluation
```

Most tests use generated documents, isolated temporary storage and mocked providers. Tests requiring `S-7200A Service Manuel.pdf` explicitly skip when that externally supplied manual is absent. Manuals, recordings, model downloads, runtime databases and private configuration are not distributed in this repository.

See [testing and research status](docs/TESTING.md) for the distinction between software regression checks and thesis evaluation. Scripts in `evaluation/` preserve development experiments; some require the local manual library, previous snapshots or labeled audio. They are not all runnable from an empty checkout. Live modes can make paid provider requests and are not part of the offline test command.

## Research status

The software is under active testing. An independently reviewed corpus, held-out engineering and Bengali evaluation, retrieval calibration, and a qualified participant-study release remain open. Source references and automated checks do not prove every generated statement. This repository does not claim verified factory performance, repair success, downtime reduction or production readiness.

## Structure

- `app/` — ingestion, structured knowledge, retrieval, diagnosis, voice, persistence and Streamlit interface.
- `tests/` — software regression checks and synthetic fixtures.
- `evaluation/` — development experiments and case definitions.
- `docs/` — current architecture and testing boundaries.
- `main.py` / `start.sh` — application entry points.

The thesis work is associated with Mahim Alam (VexP), a final-year, final-semester Industrial and Production Engineering student at RUET.
