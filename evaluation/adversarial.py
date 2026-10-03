"""Source-checked development challenges; live requests share the US$20 guard."""

import argparse
import json
import re
import time
from pathlib import Path

from app.config import Settings
from app.db import Database, utc_now
from app.language import normalize_question
from app.retrieval import Retriever
from app.service import ask
from app.structured import atomic_json
from evaluation.budget import BudgetedProvider, RunBudget


class OfflineProvider:
    def __init__(self, query): self.query = query
    def search_query(self, question): return self.query
    def answer(self, question, sources): return 'সিন্থেটিক পরীক্ষা।', [1]


def evaluate(output, live=False, selected=None, max_new_requests=0):
    settings = Settings.load()
    db = Database(settings.runtime / 'study.sqlite3')
    r = Retriever(db, settings.runtime, settings.embedding_model,
                  'cross-encoder/ms-marco-MiniLM-L6-v2')
    r.sync()
    manuals = {m['filename']:m for m in db.manuals() if m['status']=='ready'}
    cases = json.loads((settings.root/'evaluation/adversarial_cases.json').read_text())
    if selected: cases = [c for c in cases if c['id'] in selected]
    ledger=settings.runtime/'evaluations/adversarial_api_ledger.json'
    starting_attempts=len(json.loads(ledger.read_text())['attempts'])if live and ledger.exists()else 0
    budget = RunBudget(settings,max_attempts=starting_attempts+max_new_requests if max_new_requests else 128) if live else None
    report = {'created_at':utc_now(), 'live':live, 'independent_test':False,
              'interpretation':'Development evidence and explicit answer checks, not independent engineering approval or universal accuracy.',
              'retrieval_release':r.release_id, 'cases':[]}
    for case in cases:
        provider = BudgetedProvider(settings,budget) if live else OfflineProvider(case['search'])
        started=time.monotonic()
        record={**case}
        try:
            scope = [manuals[name]['id'] for name in case['manuals']] if case.get('manuals') else None
            result = ask(db,r,provider,case['question'],scope)
            evidence = result['retrieved_sources']
            joined=' '.join(normalize_question(s['text']).lower() for s in evidence)
            record.update({k:result[k] for k in ('status','answer','search_question','timings')})
            record['query_id']=result['id']
            record['evidence']=evidence
            record['evidence_check'] = (result['status']==case['status'] if case['status']=='clarify' else
                all(re.search(pattern,joined,re.I|re.S) for pattern in case.get('evidence_patterns',[])) and
                (case['page'] is None or any(s['page']==case['page'] for s in evidence)))
            if case['status']=='refused':
                record['evidence_check']=None  # Retrieval overlap cannot establish unsupportedness.
            record['answer_check'] = (result['status'] in [case['status']] and
                all(re.search(pattern,normalize_question(result['answer']),re.I|re.S)
                    for pattern in case.get('answer_patterns',[]))) if live else None
            record['error']=result.get('error')
            if not live:
                with db.connect()as c: c.execute('DELETE FROM queries WHERE id=?',(result['id'],))
        except Exception as exc:
            record.update(error=str(exc), evidence_check=False, answer_check=False if live else None)
        record['usage']=getattr(provider,'usage_log',[])
        record['raw_responses']=getattr(provider,'raw_responses',[])
        record['elapsed_seconds']=round(time.monotonic()-started,3)
        report['cases'].append(record)
        atomic_json(output,report)
        print(case['id'],record.get('status'),record['evidence_check'],record['answer_check'],record['error'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--case',action='append')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-new-requests',type=int,default=0,help='Explicit bounded continuation of the same US$20 ledger; default preserves the original request cap.')
    args=parser.parse_args()
    evaluate(args.output,args.live,args.case,args.max_new_requests)
