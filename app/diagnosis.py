"""Local conditional solver for explicit, recognised manual setting effects."""

from app.grounding import validate_diagnostic,safety_notes
from app.questions import event_trigger
from app.rules import setting_rules,observed_effect


def source_rule_answer(sources,english=False,question=''):
    if any(s.get('unstructured_observations')for s in sources):return None
    queries=next((s['retrieval_queries']for s in sources if s.get('retrieval_queries')),[])
    if not queries:
        return None
    known=next((s['known_settings']for s in sources if s.get('known_settings')), {})
    claims=[];constraints={};exclusions={}
    for intent,query in enumerate(queries):
        candidates=[]
        for number,source in enumerate(sources,1):
            if intent not in source.get('primary_intents',[])or not(source.get('setting_id')or'').startswith('function:'):
                continue
            for rule in setting_rules(source):
                if not rule['effect']or rule['trigger']!=event_trigger(query):continue
                if observed_effect(query,rule['effect'])!=rule['enabled']:continue
                settings={source['setting_id']:rule['value']}
                required={r['setting_id']:r['value']for r in source.get('setting_requirements',[])}
                required.update({r['setting_id']:r['value']for r in rule.get('conditions',[])})
                excluded={r['setting_id']:r['value']for r in rule.get('exclusions',[])}
                # Infer a complement only from both printed ON/OFF branches.
                for owner,value in excluded.items():
                    states={s.get('setting_value')for s in sources if s.get('setting_id')==owner}
                    if states=={'ON','OFF'}:required[owner]=next(iter(states-{value}))
                rejected=any(o in known and known[o]!=v for o,v in {**settings,**required}.items())or any(known.get(o)==v for o,v in excluded.items())
                for owner,value in required.items():
                    witness=next((i for i,s in enumerate(sources,1)if s.get('setting_id')==owner and s.get('setting_value')==value),None)
                    if witness:settings[owner]=value
                ids=[number]+[i for i,s in enumerate(sources,1)if s.get('setting_id')in settings and
                    s.get('setting_value')==settings[s['setting_id']]and s.get('setting_value')is not None]
                candidates.append({'text':'Source-supported conditional explanation.'if english else'ম্যানুয়ালের শর্ত অনুযায়ী সম্ভাব্য ব্যাখ্যা।',
                    'citations':list(dict.fromkeys(ids)),'intent_ids':[intent],'assessment':'ruled_out'if rejected else'possible',
                    'settings':[{'setting_id':o,'value':v}for o,v in settings.items()]})
                if not rejected:
                    for o,v in {**settings,**required}.items():constraints.setdefault(o,set()).add(v)
                    for o,v in excluded.items():exclusions.setdefault(o,set()).add(v)
        # Ambiguous rule ownership requires the general pipeline, not a guess.
        owners={s['setting_id']for c in candidates for s in c['settings']if s['setting_id'].startswith('function:')}
        if len(owners)!=1:return None
        claims.extend(candidates)
    incompatible=any(len(v)>1 or v.intersection(exclusions.get(o,set()))for o,v in constraints.items())
    parsed={'claims':claims,'citations':sorted({i for c in claims for i in c['citations']}),
            'diagnosis':'incompatible'if incompatible else'consistent'}
    try:
        answer,status=validate_diagnostic(parsed,sources,english,question)
    except ValueError:
        return None
    warning=safety_notes([s for i,s in enumerate(sources,1)if i in parsed['citations']],english)
    if warning:answer+='\n\n'+warning
    return answer,parsed['citations'],'answered'if status=='consistent'else'clarify'
