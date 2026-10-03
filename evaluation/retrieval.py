"""Offline real-manual retrieval regression; no generation or image API calls."""

import argparse
import importlib.util
import json
import sqlite3
import statistics
import time
from pathlib import Path

from app.config import Settings
from app.db import Database, utc_now
from app.retrieval import Retriever, encoder, reranker
from app.structured import atomic_json


class BaselineRows:
    def __init__(self, path):
        self.path = path

    def chunks(self, manual_ids=None):
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            query = "SELECT c.*,m.title FROM chunks c JOIN manuals m ON m.id=c.manual_id WHERE m.status='ready'"
            args = []
            if manual_ids:
                query += " AND c.manual_id IN (" + ",".join("?" for _ in manual_ids) + ")"
                args = list(manual_ids)
            return connection.execute(query, args).fetchall()


def evaluate(output_path, reference_path=None):
    settings = Settings.load()
    db = Database(settings.runtime / "study.sqlite3")
    reference_path = reference_path or settings.root / "evaluation/retrieval_cases.json"
    references = json.loads(reference_path.read_text())
    cases = {case['id']: case for case in json.loads((settings.root / 'evaluation/cases.json').read_text())}
    manuals = {row['filename']: row for row in db.manuals() if row['status'] == 'ready'}
    structured = Retriever(db, settings.runtime, settings.embedding_model)
    structured.sync()
    ranked = Retriever(db, settings.runtime, settings.embedding_model,
                       'cross-encoder/ms-marco-MiniLM-L6-v2')
    archive = settings.root / 'archive/2026-09-27-before-structured-ingestion/app/retrieval.py'
    backup = settings.runtime / 'backups/before-structured-ingestion.sqlite3'
    if not archive.exists() or not backup.exists():
        raise ValueError('The archived window baseline and its SQLite backup are required for this comparison')
    spec = importlib.util.spec_from_file_location('window_baseline', archive)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.encoder = encoder
    baseline = module.Retriever(BaselineRows(backup),
                                settings.runtime, settings.embedding_model)
    if not baseline.collection.count():
        raise ValueError('The preserved window-baseline vector collection is missing')
    structured._encode(['Unrelated synthetic warm-up question.'])
    reranker(ranked.reranker_model).predict([('Synthetic warm-up', 'Unrelated fixture.')])
    records = []
    for old in references['cases']:
        if old.get('expect_refusal'):
            continue  # Retrieval alone cannot validate model refusal.
        case = cases[old['id']]
        manual_id = manuals[case['manual']]['id']
        for label, engine in [('window_baseline', baseline), ('structured_hybrid', structured),
                              ('structured_hybrid_reranked', ranked)]:
            started = time.monotonic()
            try:
                sources = engine.search(old['search_question'], [manual_id])
                primary = [source['page'] for source in sources if not source.get('dependency')]
                all_pages = [source['page'] for source in sources]
                expected = set(case.get('expected_pages', []))
                result = {'primary_pages': primary, 'expanded_pages': sorted(set(all_pages)),
                          'primary_source_page_check': bool(expected & set(primary)),
                          'expanded_source_page_check': bool(expected & set(all_pages)),
                          'sources': sources, 'error': None}
            except ValueError as exc:
                result = {'primary_pages': [], 'expanded_pages': [], 'primary_source_page_check': False,
                          'expanded_source_page_check': False, 'sources': [], 'error': str(exc)}
            records.append({'case_id': old['id'], 'configuration': label,
                            'question_bn': case['question_bn'], 'search_question': old['search_question'],
                            'expected_pages': case.get('expected_pages', []), **result,
                            'elapsed_seconds': round(time.monotonic() - started, 3)})
            print(label, old['id'], result['primary_source_page_check'], result['error'], flush=True)
    summaries = {}
    for label in ('window_baseline', 'structured_hybrid', 'structured_hybrid_reranked'):
        selected = [row for row in records if row['configuration'] == label]
        summaries[label] = {'cases': len(selected), 'primary_page_checks': sum(row['primary_source_page_check'] for row in selected),
                            'expanded_page_checks': sum(row['expanded_source_page_check'] for row in selected),
                            'errors': sum(row['error'] is not None for row in selected),
                            'median_seconds': round(statistics.median(row['elapsed_seconds'] for row in selected), 3)}
    report = {'created_at': utc_now(), 'suite': 'development retrieval regression',
              'independent_test': False, 'new_paid_calls': 0,
              'interpretation': 'Page overlap is evidence-location coverage, not extraction/answer accuracy. Existing prototype cases are development material, not held-out thesis data. Refusal and Bengali translation were not rerun.',
              'embedding_model': settings.embedding_model, 'reranker': ranked.reranker_model,
              'reranker_max_tokens': 512, 'candidate_limit': 24,
              'manuals': [{'filename': row['filename'], 'sha256': row['sha256'], 'knowledge_release': row['knowledge_release']} for row in manuals.values()],
              'summaries': summaries, 'cases': records}
    atomic_json(output_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runtime/evaluations/structured_retrieval.json'))
    args = parser.parse_args()
    report = evaluate(args.output)
    print(json.dumps(report['summaries'], indent=2))
