"""Replay speech-era failures against real manuals; live text calls are opt-in."""

import argparse
import json

from app.config import Settings
from app.db import Database, utc_now
from app.retrieval import Retriever
from app.service import ask
from app.structured import atomic_json
from evaluation.budget import BudgetedProvider, RunBudget


CASES = [
    ('thread_breakage', 'S-7200A: The upper and lower threads keep breaking. What checks does the manual list?', 'answered', {61}),
    ('thread_and_noise', 'S-7200A মেশিনে সুতা বারবার ছিঁড়ে যাচ্ছে এবং অনেক আওয়াজ করছে। কী পরীক্ষা করব?', 'clarify', {61, 4}),
    ('missing_model', 'আমার মেশিনের সুতা বারবার ছিঁড়ে যাচ্ছে এবং অনেক সাউন্ড করছে। কী পরীক্ষা করব?', 'clarify', set()),
    ('solenoid_prerequisite', 'S-7200A প্রেসার ফুট লিফটার সবচেয়ে দ্রুত কাজ করুক, বেশি শব্দ হলেও সমস্যা নেই। কোন ফাংশনে কী মান এবং কোন পূর্বশর্ত লাগবে?', 'answered', {15}),
    ('compound_modes', 'S-7200A: After thread trimming the presser foot drops when the pedal returns to neutral. After using the knee switch, the pedal can no longer raise the foot while the machine is stopped. Which settings explain these symptoms?', 'clarify', {11, 13, 16}),
    ('subclass_exception', 'S-7200A-453: After running for 8 seconds, exactly how wide in mm should the rotary hook spattered oil band be?', 'not_applicable', {38}),
    ('second_manual', 'S-7300A: The upper and lower threads are breaking. What checks does the manual list?', 'answered', {104}),
]


def run(live=False):
    settings = Settings.load()
    db = Database(settings.runtime / 'study.sqlite3')
    retriever = Retriever(db, settings.runtime, settings.embedding_model,
                          'cross-encoder/ms-marco-MiniLM-L6-v2')
    retriever.sync()
    report = {'created_at': utc_now(), 'live': live, 'cases': [],
              'limits': ['Development regression, not independent validation',
                         'Page coverage does not prove semantic answer correctness',
                         'No user microphone recordings evaluated', 'No new PDF image calls']}
    output = settings.runtime / 'evaluations' / ('speech_rag_repair.json' if live else 'speech_rag_repair_retrieval.json')
    budget = RunBudget(settings, max_attempts=181) if live else None
    for name, question, expected_status, expected_pages in CASES:
        if live:
            provider = BudgetedProvider(settings, budget)
            result = ask(db, retriever, provider, question)
            evidence = result['retrieved_sources']
            entry = {'name': name, 'result': result, 'usage': provider.usage_log,
                     'new_api_calls': provider.attempt_count, 'expected_status': expected_status,
                     'status_matched': result['status'] == expected_status}
        else:
            # Known English query isolates retrieval from translation/generation.
            query = {'thread_and_noise': 'S-7200A: The threads are breaking and machine produces abnormal noise.'}.get(name, question)
            if any('\u0980' <= c <= '\u09ff' for c in query):
                continue
            requested = 'S-7300A' if name == 'second_manual' else 'S-7200A'
            ids = [m['id'] for m in db.manuals() if requested in m['title']]
            evidence = retriever.search(query, ids)
            entry = {'name': name, 'question': question, 'search_question': query, 'sources': evidence}
        pages = {e['page'] for e in evidence if not e.get('dependency')}
        entry.update(expected_primary_pages=sorted(expected_pages), primary_pages=sorted(pages),
                     page_check=expected_pages.issubset(pages))
        report['cases'].append(entry)
        atomic_json(output, report)
        print(name, entry.get('result', {}).get('status', 'retrieved'), 'pages', sorted(pages), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    run(parser.parse_args().live)
