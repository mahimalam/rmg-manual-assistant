"""Conservative source-derived setting branches; unknown effects stay unknown."""

import re

from app.language import normalize_question


def effect(text, context='', source=False):
    text=' '.join(text.lower().replace('’',"'").split())
    for before,after in (("can't",'cannot'),("doesn't",'does not'),("don't",'do not'),("won't",'will not'),("isn't",'is not')):
        text=text.replace(before,after)
    context=(context+' '+text).lower()
    if source and not re.search(r'\bfoot\b',text):
        return None,None
    if not re.search(r'\bfoot\b|ফুট',context):
        return None,None
    neutral_allowed=not source or bool(re.search(r'\bfoot\s+(?:(?:does|will)\s+not\s+drop|drops?|is\s+(?:not\s+)?lowered|is\s+kept\s+raised)\b',text))
    if neutral_allowed and ('neutral'in context or 'নিউট্রাল'in context):
        if re.search(r'\b(?:drops?|lowered)\b|নিচে.*(?:পড়|পড়ে|পড়ে|নাম)',text):
            negative=bool(re.search(r'(?:does|will|do)\s+not\s+drop|(?:never|no\s+longer)\s+drop|not\s+(?:drop|lowered)|(?:পড়|পড়ে|পড়ে|নাম)\S*\s+না',text))
            return 'neutral_drop',not negative
        if 'kept raised'in text:
            return 'neutral_drop',False
    operator=text if source else context
    movement=r'\brais\w*\s+and\s+lower\w*\b'if source else r'\braise[ds]?\b|\braising\b|\blower(?:ed|s|ing)?\b|\blift(?:ed|s|ing)?\b|উপরে|তোলা|ওঠানো|ওঠে|উঠ'
    if re.search(r'\btreadle\b|\bpedal\b|প্যাডেল',operator)and re.search(movement,text):
        negative=bool(re.search(r'(?:can\s*not|cannot|unable\s+to|no\s+longer|fails?\s+to|stops?|never|(?:does|do|will|is)\s+not)\s+(?:(?:be|being|able|used|to)\s+)*(?:rais\w*|lower\w*|lift\w*)\b|না(?:!|।|\s|$)',text))
        if negative:return 'treadle_operation',False
        positive=source or bool(re.search(r'\b(?:can|able\s+to|allows?|permits?|raises|lowers|lifts)\b|\bis\s+(?:raising|lowering|lifting)\b|তোলা\s+যায়|ওঠানো\s+যায়|ওঠে|উঠছে',text))
        return ('treadle_operation',True)if positive else(None,None)
    return None,None


def effect_description(name, enabled, english=False):
    if name=='neutral_drop':
        return ('The presser foot drops at neutral.' if enabled else 'The presser foot does not drop at neutral.')if english else (
            'নিউট্রালে প্রেসার ফুট নিচে নামে।'if enabled else'নিউট্রালে প্রেসার ফুট নিচে নামে না।')
    if name=='treadle_operation':
        return ('The treadle can raise/lower the presser foot.'if enabled else'The treadle cannot raise/lower the presser foot.')if english else (
            'প্যাডেল দিয়ে প্রেসার ফুট ওঠানো/নামানো যায়।'if enabled else'প্যাডেল দিয়ে প্রেসার ফুট ওঠানো/নামানো যায় না।')
    return None


def setting_rules(source):
    if 'setting_rules'in source:
        return source['setting_rules']
    if not source.get('setting_id'):
        return []
    text=source['text'].split('\nFootnote:',1)[0].split('\nResolved setting branch ',1)[0]
    if source.get('setting_value')is not None:
        match=re.search(r'(?:^|;|\n)\s*Operation:\s*(.*)',text,re.S)
        branches=[(source['setting_value'],match[1].strip()if match else text)]
        context=text
    else:
        matches=list(re.finditer(r'(?:^|\n)\s*(-?\d+(?:\.\d+)?)\s*:\s*',text))
        branches=[(m[1],text[m.end():matches[i+1].start()if i+1<len(matches)else len(text)].strip())for i,m in enumerate(matches)]
        context=text[:matches[0].start()]if matches else text
    rules=[]
    for value,quote in branches:
        name,enabled=effect(quote,context,source=True)
        if re.fullmatch(r'(?:Above|The above) operation is (?:not )?possible\.?',quote,re.I):
            previous={r['effect']for r in rules if r['effect']}
            if len(previous)==1:
                name=previous.pop();enabled='not'not in quote.lower().split()
        trigger=('after_knee_switch'if re.search(r'\bknee\b',context,re.I)else
                 'neutral_after_thread_trim'if re.search(r'neutral',context,re.I)and re.search(r'thread\s*trimm',context,re.I)else None)
        excluded=[{'setting_id':e['setting_id'],'value':e['value']}for e in source.get('setting_exclusions',[])if e['when_value']==value]
        rules.append({'value':value,'source_quote':quote,'effect':name,'enabled':enabled,
                      'trigger':trigger,'description':effect_description(name,enabled,True)or quote,
                      'conditions':[],'exclusions':excluded})
        # These exclusions were established by an exact positive/negative
        # predicate pair in the numbered note, not inferred from a range.
        notes=source.get('footnotes',[])+source['text'].split('\nFootnote:')[1:]
        for exclusion in excluded:
            note=next((n.split('\nResolved setting branch ',1)[0].strip()for n in notes if
                re.search(r'not\s+drop',n,re.I)and exclusion['value']in n and
                re.search(r'DIP\s+switch\s+'+re.escape(exclusion['setting_id'].split(':')[1])+r'\b',n,re.I)),None)
            if name=='neutral_drop'and enabled is True and note:
                rules.append({'value':value,'source_quote':note,'effect':name,'enabled':False,
                    'trigger':trigger,'description':effect_description(name,False,True),
                    'conditions':[exclusion],'exclusions':[]})
    return rules


def observed_effect(query, name):
    # Historical behavior is a different observation, not a second current state.
    query=re.split(r'\bPreviously\b|\bBefore\b|আগে(?:\s|$)',normalize_question(query),maxsplit=1,flags=re.I)[0]
    if re.search(r'sometimes|intermittent|occasionally|not always|মাঝে মাঝে|কখনো কখনো',query,re.I):return None
    actual,state=effect(query)
    return state if actual==name else None


def setting_label(owner):
    kind,number=owner.split(':',1)
    return ('Function No. 'if kind=='function'else'DIP switch ')+number


def verification_questions(sources, english=False, owners=None):
    choices={s['setting_id']for s in sources if s.get('setting_id')and not s.get('dependency')}
    prerequisites={r['setting_id']for s in sources for r in s.get('setting_requirements',[])}
    owners=set(owners)if owners is not None else choices|prerequisites
    owners|={r['setting_id']for s in sources if s.get('setting_id')in owners for r in s.get('setting_requirements',[])}
    known=next((s['known_settings']for s in sources if s.get('known_settings')), {})
    owners-=set(known)
    labels=[setting_label(o)for o in sorted(owners,key=lambda o:(o not in prerequisites,o.split(':')[0],int(o.split(':')[1])))]
    prefix='Report the current verified values, without changing them:'if english else'সেটিং পরিবর্তন না করে বর্তমান যাচাইকৃত মানগুলো জানান:'
    questions=[f'{label} = ?'for label in labels]
    queries=next((s.get('retrieval_queries')for s in sources if s.get('retrieval_queries')),[])
    if len(queries)>1:
        questions.append('Do all symptoms occur with the same settings, and in what order?'if english else'সব উপসর্গ কি একই সেটিং অবস্থায় ঘটে? কোনটির পরে কোনটি ঘটে?')
    return (prefix+'\n\n'if labels else'')+'\n'.join('- '+question for question in questions)
