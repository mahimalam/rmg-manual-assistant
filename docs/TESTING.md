# Testing and research status

This repository is a research prototype under active testing. It is intended for industrial manual access, not presented as a qualified factory deployment.

## Offline software checks

Run `uv run pytest -q`. Synthetic fixtures and mocked providers exercise ingestion, structured tables, conditional dependencies, diagnostic branch polarity, recording state, translation preservation, spending reservations and persistent query behavior.

Two fixtures additionally inspect the externally supplied `S-7200A Service Manuel.pdf`. Their dependent tests skip if that exact file is absent from the repository root. The manual is not distributed. A skip is not a passing real-manual validation.

Run `uv run python -m compileall -q app tests evaluation` to check Python syntax. Neither this command nor the regression suite proves engineering accuracy, user benefit or production readiness.

### Initial publication check — 3 October 2026

The clean publication copy passed **240 tests**, with **7 real-manual-dependent tests skipped** because the external manual is not included. Python compilation also passed. These checks used the existing local development Python environment against the publication copy; a fresh dependency installation and independent domain evaluation were not performed by this check. The application source matches the local development source; publication changes only make the two external-manual fixtures skip explicitly when their input is absent.

## Development experiments

The `evaluation/` scripts preserve development cases and replay tools. Real-manual experiments require a populated local library. Speech experiments may require labeled audio or downloaded models. The historical retrieval comparison requires its prior baseline code/database/vector snapshot; those private development artifacts are not bundled here. Scripts that replay saved failures also require their recorded local development inputs.

Live modes use the configured providers and can incur charges. Paid evaluation, new model downloads and participant collection are separate actions from the offline regression command.

## Remaining research work

- Independently reviewed manual corpus and reference answers.
- Held-out retrieval, answer and Bengali speech evaluation.
- Defined acceptance thresholds and retrieval/refusal calibration.
- Target-hardware and intended-user evaluation.
- Qualified study protocol and participant-study release.

Claims about factory use, diagnosis accuracy, repair success, reduced downtime or research outcomes require evidence from the corresponding evaluation. None is asserted by this initial publication.
