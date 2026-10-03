"""Source-checked diagnostic cases and exact request audit; paid mode is opt-in."""

import argparse
import hashlib
import json
from pathlib import Path

from app.config import Settings
from app.db import Database,utc_now
from app.provider import ManualProvider,_answer_json
from app.retrieval import Retriever,encoder
from app.service import ask,continue_diagnosis
from app.structured import atomic_json
from evaluation.budget import BudgetedProvider,RunBudget


def expected(sources,kind):
    def number(owner,value=None):
        return next(i for i,s in enumerate(sources,1)if s.get('setting_id')==owner and
                    (value is None or s.get('setting_value')==value))
    def claim(owner,value,intent,state,assessment='possible'):
        row=number(owner);dip=number('dip:1',state)
        return {'text':'ম্যানুয়ালের এই শাখার শর্ত যাচাই করুন।','citations':[row,dip],
                'intent_ids':[intent],'assessment':assessment,
                'settings':[{'setting_id':owner,'value':value},{'setting_id':'dip:1','value':state}]}
    if kind in ('compound','no-longer-compound'):claims=[claim('function:12','0',0,'ON'),claim('function:40','0',1,'OFF')];diagnosis='incompatible'
    else:
        claims=[claim('function:40','1'if kind=='positive'else'0',0,'OFF','ruled_out'if kind=='excluded'else'possible')]
        diagnosis='insufficient'if kind=='excluded'else'consistent'
    return {'answer':'ব্যাখ্যা।','supported':True,'citations':sorted({i for c in claims for i in c['citations']}),
            'claims':claims,'diagnosis':diagnosis}


def evaluate(output,live=False):
    s=Settings.load();db=Database(s.runtime/'study.sqlite3')
    r=Retriever(db,s.runtime,s.embedding_model,'cross-encoder/ms-marco-MiniLM-L6-v2');r.sync()
    manual=next(m for m in db.manuals()if m['filename']=='S-7200A Service Manuel.pdf')
    original=json.loads((s.runtime/'evaluations/knee_switch_root_cause.json').read_text())['original_record']
    cases=[('compound',original['question'],original['search_question'],'clarify'),
           ('negative','On Brother S-7200A, after knee switch use while stopped, the treadle cannot raise the presser foot. Which function value and enabling DIP state explain this?',None,'answered'),
           ('positive','On Brother S-7200A, after knee switch use while stopped, the treadle can raise and lower the presser foot. Which Function 40 value and enabling DIP state permit this?',None,'answered'),
           ('confirmed','Function 40 = 0; DIP switch 1 = OFF',None,'answered'),
           ('excluded','Function 40 = 1; DIP switch 1 = OFF',None,'clarify')]
    counterexample=s.runtime/'evaluations/diagnosis_user_counterexample_before.json'
    if counterexample.exists():
        saved=json.loads(counterexample.read_text())['query']
        cases.append(('no-longer-compound',saved['question'],saved['search_question'],'clarify'))
    budget=None
    if live:
        count=len(json.loads((s.runtime/'evaluations/adversarial_api_ledger.json').read_text())['attempts'])
        budget=RunBudget(s,max_attempts=count+8)
    report={'created_at':utc_now(),'live':live,'model':s.model,'release':r.release_id,
            'independent_engineering_validation':False,'labels':'Provisional source-reviewed development cases; not held-out thesis scores.',
            'cases':[],'new_image_calls':0}
    parent=None;temporary=[]
    try:
        for kind,question,search,status in cases:
            payloads=[]
            if live:
                provider=BudgetedProvider(s,budget)
                post=provider._post_response
                def captured(payload):payloads.append(payload);return post(payload)
                provider._post_response=captured
            else:
                provider=ManualProvider('https://example.test','example','synthetic')
                provider.search_query=lambda q:search or q
                def capture(payload):
                    payloads.append(payload)
                    content=json.dumps(expected(provider.evidence,kind),ensure_ascii=False)
                    provider.raw_responses.append({'content':content})
                    return {'content':content,'finish_reason':'stop'}
                provider._post_response=capture
            generate=provider.answer
            def answer(q,sources):provider.evidence=sources;return generate(q,sources)
            provider.answer=answer
            result=(continue_diagnosis(db,r,provider,parent,question)if kind in ('confirmed','excluded')else
                    ask(db,r,provider,question,[manual['id']]))
            temporary.append(result['id'])
            if kind=='negative':parent=result['id']
            generation=[p for p in payloads if p.get('response_format')]
            audit_mode='actual generation request'
            if not generation:
                audit=ManualProvider('https://example.test','example','synthetic')
                def witness(payload):
                    generation.append(payload)
                    return {'content':'{"answer":"অপর্যাপ্ত তথ্য।","supported":false,"citations":[]}', 'finish_reason':'stop'}
                audit._post_response=witness;audit.answer(question,result['retrieved_sources'])
                audit_mode='synthetic payload audit; source-rule path did not request model generation'
            packet=generation[-1]['messages'][1]['content']
            all_sent=bool(packet)and all(e['text']in packet for e in result['retrieved_sources'])
            check=result['status']==status and 'Function No. 40' in result['answer']
            if kind in ('compound','no-longer-compound'):
                check=check and 'Function No. 12 = 0'in result['answer']and 'Function No. 40 = 0'in result['answer']and 'Function No. 40 = 1'not in result['answer']
                check=check and 'প্যাডেল দিয়ে প্রেসার ফুট ওঠানো/নামানো যায় না।'in result['answer']and '**'in result['answer']
            if kind=='positive':check=check and 'Function No. 40 = 1'in result['answer']and 'ওঠানো/নামানো যায়।'in result['answer']
            if kind in ('negative','confirmed'):check=check and 'Function No. 40 = 0'in result['answer']and 'ওঠানো/নামানো যায় না।'in result['answer']
            if kind=='excluded':check=check and 'বাদ'in result['answer']and 'আগে, পরে'in result['answer']
            model_check=bool(check and not result.get('error'))if result.get('reasoning_path')!='source_rules'else None
            record={'id':kind,'question':question,'expected_status':status,'result':result,
                    'production_check':bool(check),'generation_check':model_check,'all_retrieved_text_sent':all_sent,
                    'payload_audit_mode':audit_mode,
                    'generation_payloads':generation,'payload_sha256':hashlib.sha256(packet.encode()).hexdigest(),
                    'raw_responses':provider.raw_responses,'usage':provider.usage_log,'new_api_calls':provider.attempt_count}
            report['cases'].append(record);atomic_json(output,report)
            print(kind,result['status'],'production=',bool(check),'generation=',model_check,'all_text_sent=',all_sent,flush=True)
        # Full corpus token audit includes cached embeddings, not only new misses.
        model=encoder(s.embedding_model);rows=db.chunks();lengths=[]
        for row in rows:lengths.append(len(model.tokenizer(row['text'],truncation=False)['input_ids']))
        report['embedding_audit']={'records':len(rows),'max_tokens':max(lengths),'model_limit':model.max_seq_length,
                                   'over_limit':sum(n>model.max_seq_length for n in lengths)}
        assert report['embedding_audit']['over_limit']==0
        report['new_api_calls']=sum(c['new_api_calls']for c in report['cases'])
        report['passed']=all(c['production_check']and c['all_retrieved_text_sent']for c in report['cases'])
        atomic_json(output,report)
        if not live:assert report['passed'],[(c['id'],c['result'].get('error'))for c in report['cases']]
    finally:
        if not live:
            with db.connect()as connection:
                for query in temporary:connection.execute('DELETE FROM queries WHERE id=?',(query,))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();evaluate(args.output,args.live)
