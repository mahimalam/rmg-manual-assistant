"""Run source-page and refusal checks with manually written evaluation cases."""

import argparse
import json
import statistics
import time
from pathlib import Path

from app.config import Settings
from app.db import Database, utc_now
from app.provider import ManualProvider, ProviderError
from app.retrieval import Retriever
from app.service import ask


def evaluate(cases_path: Path, output_path: Path, runtime: Path | None = None, use_reranker=True):
    settings = Settings.load()
    runtime = runtime or settings.runtime
    db = Database(runtime / "study.sqlite3")
    retriever = Retriever(db, runtime, settings.embedding_model,
                          "cross-encoder/ms-marco-MiniLM-L6-v2" if use_reranker else None)
    retriever.sync()
    provider = ManualProvider(settings.base_url, settings.api_key, settings.model)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    manuals = {row["filename"]: row for row in db.manuals() if row["status"] == "ready"}
    results = []
    for case in cases:
        filename = case["manual"]
        if filename not in manuals:
            raise ValueError(f"Evaluation manual is not loaded: {filename}")
        started = time.monotonic()
        try:
            result = ask(db, retriever, provider, case["question_bn"],
                         [manuals[filename]["id"]])
            cited_pages = [source["page"] for source in result["sources"]]
            expected_pages = case.get("expected_pages", [])
            verdict = (
                result["status"] in ("no_evidence", "refused")
                if case.get("expect_refusal")
                else bool(set(cited_pages) & set(expected_pages))
            )
            results.append({"id": case["id"], "status": result["status"],
                            "expected_facts": case.get("expected_facts", []),
                            "expect_refusal": bool(case.get("expect_refusal")),
                            "search_question": result["search_question"],
                            "cited_pages": cited_pages, "expected_pages": expected_pages,
                            "source_page_check": verdict, "answer": result["answer"],
                            "timings": result["timings"], "sources": result["sources"],
                            "error": None,
                            "elapsed_seconds": round(time.monotonic() - started, 2)})
        except (ProviderError, ValueError) as exc:
            results.append({"id": case["id"], "status": "error", "error": str(exc),
                            "expected_facts": case.get("expected_facts", []),
                            "expect_refusal": bool(case.get("expect_refusal")),
                            "source_page_check": False,
                            "elapsed_seconds": round(time.monotonic() - started, 2)})
    times = [row["elapsed_seconds"] for row in results]
    report = {
        "created_at_utc": utc_now(), "model": settings.model,
        "embedding_model": settings.embedding_model,
        "retrieval_release": retriever.release_id, "reranker": retriever.reranker_model,
        "manuals": [{"filename": row["filename"], "sha256": row["sha256"],
                     "pages": row["pages"], "knowledge_release": row["knowledge_release"]}
                    for row in manuals.values()],
        "case_count": len(results),
        "source_page_checks_passed": sum(bool(row["source_page_check"]) for row in results),
        "median_elapsed_seconds": round(statistics.median(times), 2) if times else None,
        "cases": results,
        "interpretation": ("A cited expected page is a retrieval/citation check, not a factual "
                           "correctness score. A qualified reviewer must grade every answer "
                           "and inspect source images, conditions, and omissions."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-reranker", action="store_true", help="Use hybrid retrieval alone")
    args = parser.parse_args()
    result = evaluate(args.cases, args.output, use_reranker=not args.no_reranker)
    print(f"{result['source_page_checks_passed']}/{result['case_count']} source-page checks passed")
    print(f"Report: {args.output}")
