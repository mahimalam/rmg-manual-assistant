"""Real-corpus regression for the reported compound question; live is opt-in."""

import argparse
import json
from pathlib import Path

import httpx
from unittest.mock import patch

from app.config import Settings
from app.db import Database, utc_now
from app.provider import ManualProvider, ProviderError
from app.retrieval import Retriever
from app.service import ask
from app.structured import atomic_json
from evaluation.budget import BudgetedProvider, RunBudget


def evaluate(output, live=False, reuse_search=False):
    s=Settings.load();db=Database(s.runtime/'study.sqlite3')
    r=Retriever(db,s.runtime,s.embedding_model,'cross-encoder/ms-marco-MiniLM-L6-v2');r.sync()
    with db.connect()as c:
        original=c.execute('SELECT * FROM queries WHERE id=?',('89e83eb23bf54180a0801e20c2e8a57c',)).fetchone()
    # The mirror retains different history; use the preserved development record.
    if original is None:
        original=json.loads((s.runtime/'evaluations/knee_switch_root_cause.json').read_text())['original_record']
    original=dict(original)
    manual=next(m for m in db.manuals()if m['filename']=='S-7200A Service Manuel.pdf')
    evidence=r.search(original['search_question'],[manual['id']])
    assert any(e.get('setting_id')=='function:12'for e in evidence)
    assert any(e.get('setting_id')=='function:40'for e in evidence)
    assert any(e['page']==11 and e.get('note_ids')==['1']for e in evidence)
    assert not any(e.get('setting_id')in ('dip:3','dip:4')for e in evidence if not e.get('dependency'))
    queries=evidence[0]['retrieval_queries']
    assert len(queries)==2
    def number(owner,value=None):
        return next(i for i,e in enumerate(evidence,1)if e.get('setting_id')==owner and
                    (value is None or e.get('setting_value')==value))
    trim,knee,on,off=number('function:12'),number('function:40'),number('dip:1','ON'),number('dip:1','OFF')
    claims=[{'text':'সুতা কাটার পর নিউট্রালে ফুট নামা Function 12 = 0 এবং DIP switch 1 = ON অবস্থার সঙ্গে মেলে।',
             'citations':[trim,on],'intent_ids':[0],
             'settings':[{'setting_id':'function:12','value':'0'},{'setting_id':'dip:1','value':'ON'}]},
            {'text':'হাঁটু সুইচ ব্যবহারের পর প্যাডেলে ফুট না ওঠা Function 40 = 0 এবং DIP switch 1 = OFF অবস্থার সঙ্গে মেলে।',
             'citations':[knee,off],'intent_ids':[1],
             'settings':[{'setting_id':'function:40','value':'0'},{'setting_id':'dip:1','value':'OFF'}]}]
    valid={'answer':'পরীক্ষার উত্তর।','supported':True,'citations':[trim,knee,on,off],
           'claims':claims,'diagnosis':'incompatible'}
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={
        'choices':[{'message':{'content':json.dumps(valid,ensure_ascii=False)}}]})))as client:
        provider=ManualProvider('https://example.test','example','synthetic',client)
        answer,ids=provider.answer(original['question'],evidence)
        assert provider.last_answer_status=='clarify'and 'আলাদা সম্ভাব্য' in answer
    bad=json.loads(original['provider_responses_json'])[-1]['content']
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={
        'choices':[{'message':{'content':bad}}]})))as client:
        try:ManualProvider('https://example.test','example','synthetic',client).answer(original['question'],evidence)
        except ProviderError as exc:rejection=str(exc)
        else:raise AssertionError('Original erroneous response was accepted')
    report={'created_at':utc_now(),'live':live,'model_comparison':False,'independent_validation':False,
            'release':r.release_id,'question':original['question'],'search_question':original['search_question'],
            'retrieval_queries':queries,'evidence':evidence,'retrieval_check':True,
            'original_response_rejected':rejection,'valid_mock_response':answer,'valid_mock_status':'clarify',
            'new_api_calls':0}
    # Replay captured failures through the production flow, without rebilling.
    replay_inputs=[('original',bad)]
    for filename in ('pipeline_live_same_model.json','pipeline_live_verified.json'):
        path=s.runtime/'evaluations'/filename
        if path.exists():
            saved=json.loads(path.read_text())
            replay_inputs.append((filename,saved['raw_responses'][-1]['content']))
    report['production_failure_replays']=[]
    for label,content in replay_inputs:
        with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={
                'choices':[{'message':{'content':content}}]})))as client:
            provider=ManualProvider('https://example.test','example','synthetic',client)
            provider.search_query=lambda question:original['search_question']
            with patch('app.service.source_rule_answer',return_value=None):
                result=ask(db,r,provider,original['question'],[manual['id']])
            try:
                assert result['status']=='clarify',result['status']
                assert result['error'] and 'Function No. 40' in result['answer']and 'Function No. 12' in result['answer']
                assert content not in result['answer']
                with db.connect()as connection:
                    record=connection.execute('SELECT error,provider_responses_json FROM queries WHERE id=?',(result['id'],)).fetchone()
                assert record['error'] and json.loads(record['provider_responses_json'])[-1]['content']==content
                report['production_failure_replays'].append({'input':label,'raw_response':content,
                    'result':result,'forced_generation_replay':True,'rejected_response_preserved':True,'new_api_calls':0})
            finally:
                with db.connect()as connection:connection.execute('DELETE FROM queries WHERE id=?',(result['id'],))
    # Exercise translation-loss fallback with real multilingual retrieval.
    report['bengali_fallback_evidence']=r.search(original['question'],[manual['id']])
    assert any(e.get('setting_id')=='function:40'for e in report['bengali_fallback_evidence'])
    assert any(e.get('setting_id')=='function:12'for e in report['bengali_fallback_evidence'])
    atomic_json(output,report)
    if live:
        provider=BudgetedProvider(s,RunBudget(s))
        if reuse_search:
            provider.search_query=lambda question:original['search_question']
            report['translation_mode']='Reused the prior complete translation; generation only is live.'
        result=ask(db,r,provider,original['question'],[manual['id']])
        report.update(live_result=result,usage=provider.usage_log,raw_responses=provider.raw_responses,
                      new_api_calls=provider.attempt_count)
        atomic_json(output,report)
        assert result['status']=='clarify',result
        assert '40' in result['answer']or '৪০'in result['answer']
        assert '12'in result['answer']or '১২'in result['answer']
        print('Live same-model result:',result['status'],result['answer'],flush=True)
    print('Pipeline regression passed; original bad response rejected; valid separated modes accepted.',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-search',action='store_true',help='Reuse the saved complete search translation; avoid another paid translation.')
    args=parser.parse_args();evaluate(args.output,args.live,args.reuse_search)
