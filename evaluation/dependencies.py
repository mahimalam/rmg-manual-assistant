"""Development evidence regression; --live explicitly adds paid text generation."""

import argparse
import json
import time
from pathlib import Path

from app.config import Settings
from app.db import Database, utc_now
from app.provider import ManualProvider, ProviderError
from app.retrieval import Retriever
from app.service import ask
from app.structured import atomic_json


def evidence_check(case, evidence):
    joined = " ".join(" ".join(source["text"].split()) for source in evidence)
    phrase = case.get("expected_source_phrase", "")
    present = not phrase or phrase in joined
    if case["kind"] == "setting":
        return (present and any(source.get("setting_id") == case["setting_id"] for source in evidence)
                and any(case["requirement"] in source.get("setting_requirements", []) for source in evidence))
    if case["kind"] == "exclusion":
        return present and any(source.get("applicability", {}).get(case["expected_subclass"]) ==
                               "explicitly_excluded" for source in evidence)
    if case["kind"] == "scope_control":
        return (present and any(source.get("applicability", {}).get(case["expected_subclass"]) ==
                                "within_declared_scope" for source in evidence)
                and not any(source.get("applicability", {}).get(case["expected_subclass"]) ==
                            "explicitly_excluded" for source in evidence))
    return None  # Retrieval alone cannot validate refusal.


def evaluate(output, live=False, case_ids=None):
    settings = Settings.load()
    db = Database(settings.runtime / "study.sqlite3")
    retriever = Retriever(db, settings.runtime, settings.embedding_model,
                          "cross-encoder/ms-marco-MiniLM-L6-v2")
    retriever.sync()
    manuals = {row["filename"]: row for row in db.manuals() if row["status"] == "ready"}
    cases = json.loads((settings.root / "evaluation/dependency_cases.json").read_text())
    if case_ids:
        unknown = set(case_ids) - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"Unknown regression cases: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in case_ids]
    records, calls, attempts = [], 0, 0
    for case in cases:
        start = time.monotonic()
        record = {"case_id": case["id"], "kind": case["kind"], "question": case["question"],
                  "expected_facts": case["expected_facts"]}
        provider = ManualProvider(settings.base_url, settings.api_key, settings.model)
        try:
            manual_id = manuals[case["manual"]]["id"]
            if live:
                result = ask(db, retriever, provider, case["question"], [manual_id])
                evidence = result["retrieved_sources"]
                record.update({key: result[key] for key in ("id", "answer", "status", "sources", "timings", "search_question")})
                record["response_shape_check"] = (result["status"] == "refused" if case["kind"] == "unsupported" else
                    result["status"] == "not_applicable" if case["kind"] == "exclusion" else result["status"] == "answered")
                record["bengali_present"] = any("\u0980" <= character <= "\u09ff" for character in result["answer"])
            else:
                evidence = retriever.search(case["search_question"], [manual_id])
            record.update(evidence=evidence, evidence_completeness_check=evidence_check(case, evidence), error=None)
        except (ValueError, ProviderError) as exc:
            record.update(error=str(exc), evidence_completeness_check=False)
        record["usage"] = provider.usage_log
        record["raw_responses"] = provider.raw_responses
        record["validation_issue"] = provider.validation_issue
        calls += len(provider.usage_log)
        attempts += provider.attempt_count
        record["elapsed_seconds"] = round(time.monotonic() - start, 3)
        records.append(record)
        print(case["id"], record.get("status", "offline"), record["evidence_completeness_check"],
              record.get("error"), flush=True)
    report = {"created_at": utc_now(), "suite": "dependency and applicability development regression",
              "live": live, "successful_text_api_calls": calls, "image_api_calls": 0,
              "text_api_attempts": attempts, "billing_certainty": "Token usage when reported; gateway tariff remains unverified",
              "independent_test": False, "retrieval_release": retriever.release_id,
              "requested_model": settings.model, "embedding_model": settings.embedding_model,
              "reranker": retriever.reranker_model,
              "manuals": [{"filename": row["filename"], "sha256": row["sha256"],
                           "knowledge_release": row["knowledge_release"]} for row in manuals.values()],
              "interpretation": "Mechanism coverage and response shape are not factual accuracy. Questions are development cases; independent domain review remains open. No answer keys were supplied to retrieval or generation.",
              "cases": records}
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make up to six paid answer calls plus Bengali query translation")
    parser.add_argument("--output", type=Path, default=Path("runtime/evaluations/dependency_retrieval.json"))
    parser.add_argument("--case", action="append", help="Run only this case ID; repeat to select multiple cases")
    options = parser.parse_args()
    evaluate(options.output, options.live, options.case)
