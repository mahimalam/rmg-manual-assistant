"""Explicit evidence and mode checks for compound answers, not an LLM judge."""

import re

from app.language import normalize_question
from app.questions import event_trigger
from app.rules import (setting_rules, observed_effect, effect_description,
                       setting_label, verification_questions)


def scope_notes(sources, english=False):
    """Retain individual source checks whose qualifiers cannot scope a whole row."""
    notes = []
    for source in sources:
        for line in source['text'].splitlines():
            if re.search(r'\([^()\n]*(?:specifications?|models?)[^()\n]*\)', line, re.I):
                notes.append(line.strip() + f" ({source['title']}, PDF p. {source['page']})")
    if not notes:
        return ''
    label = 'Conditions on the individual checks (source wording)' if english else 'চেকগুলোর প্রযোজ্যতার শর্ত (ম্যানুয়ালের মূল ভাষায়)'
    return '**' + label + '**\n\n' + '\n'.join('- ' + note for note in dict.fromkeys(notes))


def safety_notes(sources, english=False):
    """Keep actual warning instructions, not duplicated section headings."""
    notes={}
    for source in sources:
        if source.get('dependency')!='warning':continue
        warnings=source.get('warnings')
        if warnings is None:
            warnings=[source['text'].split('Warning:',1)[-1]]
        for warning in warnings:
            text=' '.join(re.sub(r'^\s*(?:DANGER|WARNING|CAUTION)\b\s*','',warning).split())
            if re.search(r'instructions which follow this term', text, re.I):
                continue
            if len(text.split())<8 and not re.search(r'\b(?:wait|turn|disconnect|unplug|never|must|avoid|keep|wear|set)\b|do not',text,re.I):continue
            notes.setdefault(text,set()).add((source['title'],source['page']))
    if not notes:return ''
    label='Source safety instructions (original language)'if english else'নিরাপত্তা নির্দেশ (ম্যানুয়ালের মূল ভাষায়)'
    return '**'+label+'**\n\n'+'\n'.join('- '+text+' ('+'; '.join(f'{title}, PDF p. {page}'for title,page in sorted(refs))+')'for text,refs in notes.items())


def source_clarification(sources, english=False):
    """Render literal source records when a generated diagnosis fails checks."""
    queries = next((s['retrieval_queries'] for s in sources if s.get('retrieval_queries')), [])
    control = not queries or any(event_trigger(q) or re.search(r'function|DIP|setting|ফাংশন|ডিপ|সেটিং', q, re.I) for q in queries)
    selected=[i for i,s in enumerate(sources,1)if not s.get('dependency') and
              (s.get('setting_id') or s.get('note_ids'))]
    if not control:
        selected = []
        for intent in range(len(queries)):
            candidates = [i for i, s in enumerate(sources, 1) if intent in s.get('primary_intents', [])
                          and not s.get('setting_id') and not s.get('note_ids')]
            if candidates:
                selected.append(candidates[0])
    if not selected:
        selected=[i for i,s in enumerate(sources,1)if not s.get('dependency')][:4]
    by_id={s.get('id'):i for i,s in enumerate(sources,1)}
    ids=list(selected)
    for i in ids:
        for key in sources[i-1].get('required_evidence_ids',[]):
            linked=by_id.get(key)
            if linked and linked not in ids:ids.append(linked)
    prefix=('The generated diagnosis failed evidence checks. The relevant manual rules below do not establish the actual settings or a single cause. Verify the settings and event sequence.' if english else
            'তৈরি করা সিদ্ধান্তটি ম্যানুয়ালের শর্তের সঙ্গে যাচাইয়ে মেলেনি, তাই সেটিং পরিবর্তনের নির্দেশ দেওয়া হচ্ছে না। নিচে প্রাসঙ্গিক মূল নিয়মগুলো দেওয়া হলো। বর্তমান সেটিং এবং উপসর্গ ঘটার ক্রম যাচাই করুন; এগুলো থেকে একক কারণ নিশ্চিত করা যায়নি।')
    if not control:
        prefix = ('A single cause has not been established. Relevant manual checks for each symptom are shown below; unavailable procedure details cannot be supplied.' if english else
                  'সব উপসর্গের একক কারণ নিশ্চিত করা যায়নি। প্রতিটি উপসর্গের জন্য ম্যানুয়ালের প্রাসঙ্গিক চেকগুলো নিচে আছে। অনির্ভরযোগ্যভাবে নিষ্কাশিত অংশ থেকে সমন্বয়ের বিস্তারিত দেওয়া যাচ্ছে না।')
    blocks=[]
    for i in selected:
        s=sources[i-1];owner=s.get('setting_id')
        label=('Function No. ' if owner.startswith('function:')else'DIP switch ')+owner.split(':')[1] if owner else 'Manual note'
        blocks.append(f"{label} ({s['title']}, PDF p. {s['page']}):\n{s['text']}")
    warning=safety_notes([sources[i-1]for i in ids],english)
    if warning:blocks.append(warning)
    checks = verification_questions([sources[i-1] for i in ids], english) if control else ''
    return '\n\n'.join(part for part in [prefix,checks,*dict.fromkeys(blocks)] if part),ids


def _value_supported(source, value):
    if source.get('setting_value') is not None:
        return source['setting_value']==value
    # Function values must be an explicit branch or lie in its printed range.
    text=normalize_question(source['text'])
    if re.search(r'(?:^|\n)\s*'+re.escape(value)+r'\s*:',text):
        return True
    interval=re.search(r'Setting range:\s*(-?\d+(?:\.\d+)?)\s*[–−-]\s*(-?\d+(?:\.\d+)?)',text)
    try:
        return bool(interval and float(interval[1])<=float(value)<=float(interval[2]))
    except ValueError:
        return False


def validate_diagnostic(parsed, sources, english=False, question=''):
    queries=next((s['retrieval_queries']for s in sources if s.get('retrieval_queries')),[])
    if len(queries)<2 and not any(s.get('require_claims')for s in sources):
        return None
    by_id={s.get('id'):i for i,s in enumerate(sources,1)}
    # A function citation also carries its required source definitions/notes.
    for claim in parsed.get('claims',[])if isinstance(parsed.get('claims'),list)else []:
        if not isinstance(claim,dict)or not isinstance(claim.get('citations'),list):continue
        for number in claim['citations']:
            if type(number)is not int or not 1<=number<=len(sources):continue
            for key in sources[number-1].get('required_evidence_ids',[]):
                if key not in by_id:raise ValueError('Diagnostic claim has a missing required source')
                linked=by_id[key]
                if linked not in claim['citations']:claim['citations'].append(linked)
                if linked not in parsed['citations']:parsed['citations'].append(linked)
    claims=parsed.get('claims')
    if not isinstance(claims,list) or not 1<=len(claims)<=12:
        raise ValueError('Compound diagnostic answer requires source-linked claims')
    diagnosis=parsed.get('diagnosis')
    # A checklist with no cited control settings cannot establish a machine
    # state. Derive that assessment locally instead of trusting an LLM enum.
    mechanical = all(isinstance(c, dict) and c.get('settings') == [] and
                     isinstance(c.get('citations'), list) and c['citations'] and
                     all(type(i) is int and 1 <= i <= len(sources) and
                         not sources[i-1].get('setting_id') and not sources[i-1].get('note_ids')
                         for i in c['citations']) for c in claims)
    if mechanical:
        diagnosis = 'insufficient'
    if diagnosis not in ('consistent','incompatible','insufficient'):
        raise ValueError('Missing diagnostic compatibility assessment')
    covered, active_covered, assignments, exclusions = set(), set(), {}, {}
    declared_by_intent={i:set()for i in range(len(queries))}
    known=next((s['known_settings']for s in sources if s.get('known_settings')), {})
    rendered=[]
    used=set(parsed['citations'])
    for claim in claims:
        if not isinstance(claim,dict) or not isinstance(claim.get('text'),str) or not claim['text'].strip():
            raise ValueError('Invalid diagnostic claim')
        ids=claim.get('citations');intents=claim.get('intent_ids');settings=claim.get('settings',[])
        if (not isinstance(ids,list) or not ids or any(type(i)is not int or i not in used for i in ids) or
            not isinstance(intents,list) or not intents or any(type(i)is not int or not 0<=i<len(queries) for i in intents) or
            not isinstance(settings,list)):
            raise ValueError('Invalid diagnostic claim citations or intent IDs')
        evidence=[sources[i-1]for i in ids]
        for intent in intents:
            if not any(intent in s.get('primary_intents',s.get('retrieval_intents',[]) if not s.get('dependency') else [])for s in evidence):
                raise ValueError('Claim has no primary evidence for its intent')
        covered.update(intents)
        text=normalize_question(claim['text'])
        if not english and not re.search(r'[\u0980-\u09ff]',text):
            raise ValueError('Diagnostic claim did not use Bengali')
        mentioned={'function:'+m for m in re.findall(r'(?:function(?:\s+(?:no|number)\.?)?|ফাংশন(?:\s+(?:নম্বর|নং))?)\s*[#:]?\s*(\d+)',text,re.I)}
        mentioned.update('dip:'+m for m in re.findall(r'(?:DIP\s+switch|ডিপ\s+সুইচ)\s*(\d+)',text,re.I))
        declared={item.get('setting_id')for item in settings if isinstance(item,dict)}
        display_owners={owner for owner in declared if isinstance(owner,str)and owner.startswith('function:')}or declared
        referenced={s.get('setting_id')for s in evidence}
        if mentioned-declared-referenced:
            raise ValueError('Diagnostic claim mentions an undeclared setting')
        for intent in intents:declared_by_intent[intent].update(declared)
        for label,owner in ((r'(?:Function(?:\s+(?:No|number)\.?)?|ফাংশন(?:\s+(?:নম্বর|নং))?)','function'),
                            (r'(?:DIP\s+switch|ডিপ\s+সুইচ)','dip')):
            for match in re.finditer(label+r'\s*(\d+)\s*=\s*(\d+(?:\.\d+)?|ON|OFF|অন|অফ)\b',text,re.I):
                value={'অন':'ON','অফ':'OFF'}.get(match[2].upper(),match[2].upper())
                if not any(item.get('setting_id')==owner+':'+match[1] and
                           normalize_question(item.get('value','')).upper()==value for item in settings):
                    raise ValueError('Diagnostic claim text contradicts its checked setting value')
        constraints,claim_exclusions={},{}
        assessment=claim.get('assessment','possible')
        if assessment not in ('possible','ruled_out'):
            raise ValueError('Invalid hypothesis assessment')
        descriptions=[]
        states={**known,**{s['setting_id']:normalize_question(s['value']).upper()for s in settings
                          if isinstance(s,dict)and isinstance(s.get('setting_id'),str)and isinstance(s.get('value'),str)}}
        for setting in settings:
            if (not isinstance(setting,dict) or not isinstance(setting.get('setting_id'),str) or
                    not isinstance(setting.get('value'),str)):
                raise ValueError('Invalid diagnostic setting claim')
            owner,value=setting['setting_id'],normalize_question(setting['value']).upper()
            owners=[s for s in evidence if s.get('setting_id')==owner]
            if not owners or not any(_value_supported(s,value)for s in owners):
                raise ValueError('Diagnostic setting value lacks its source definition')
            if not any(set(intents).intersection(s.get('retrieval_intents',[]))for s in owners):
                raise ValueError('Diagnostic setting evidence belongs to a different intent')
            constraints.setdefault(owner,set()).add(value)
            all_rules=[rule for source in owners for rule in setting_rules(source)if rule['value']==value]
            rules=[rule for rule in all_rules if all(states.get(c['setting_id'])==c['value']for c in rule.get('conditions',[]))and
                   not any(states.get(c['setting_id'])==c['value']for c in rule.get('exclusions',[]))]
            if not rules and assessment=='ruled_out':rules=[rule for rule in all_rules if not rule.get('conditions')]
            if all_rules and not rules:raise ValueError('Setting effect has incompatible or unmet enabling conditions')
            for rule in rules:
                for condition in rule.get('conditions',[]):constraints.setdefault(condition['setting_id'],set()).add(condition['value'])
                if rule['effect']:
                    observed={observed_effect(queries[i],rule['effect'])for i in intents}-{None}
                    if assessment!='ruled_out'and observed and observed!={rule['enabled']}:
                        raise ValueError('Setting branch effect contradicts the requested symptom')
                    description=effect_description(rule['effect'],rule['enabled'],english)
                    if rule['trigger']=='after_knee_switch':
                        description=('After knee-switch use: 'if english else'হাঁটু সুইচ ব্যবহারের পরে: ')+description
                    elif rule['trigger']=='neutral_after_thread_trim':
                        description=('After thread trimming: 'if english else'সুতা কাটার পরে: ')+description
                else:
                    description=('Source branch: 'if english else'ম্যানুয়ালের মূল শাখা: ')+rule['source_quote']
                if owner in display_owners and (not rule['trigger']or any(event_trigger(queries[i])==rule['trigger']for i in intents)):
                    descriptions.append(f"{setting_label(owner)} = {value}: {description}")
            for source in owners:
                for requirement in source.get('setting_requirements',[]):
                    constraints.setdefault(requirement['setting_id'],set()).add(requirement['value'])
                for exclusion in source.get('setting_exclusions',[]):
                    if exclusion['when_value']==value and not any(r.get('conditions')for r in rules):
                        claim_exclusions.setdefault(exclusion['setting_id'],set()).add(exclusion['value'])
        if any(len(values)>1 or values.intersection(claim_exclusions.get(key,set())) for key,values in constraints.items()):
            raise ValueError('A diagnostic claim combines incompatible setting requirements')
        if assessment!='ruled_out':
            if any(key in known and known[key]not in values for key,values in constraints.items())or any(
                    known.get(key)in values for key,values in claim_exclusions.items()):
                raise ValueError('Hypothesis conflicts with the observed current settings')
            active_covered.update(intents)
            for key,values in constraints.items():assignments.setdefault(key,set()).update(values)
            for key,values in claim_exclusions.items():exclusions.setdefault(key,set()).update(values)
        if descriptions:
            label=('Excluded candidate'if assessment=='ruled_out'else'Possible setting explanation')if english else(
                'এই সম্ভাবনাটি বাদ পড়ে'if assessment=='ruled_out'else'সম্ভাব্য সেটিংয়ের ব্যাখ্যা')
            requirements=[f"{setting_label(key)} = {next(iter(values))}"for key,values in constraints.items()if key not in display_owners]
            details='\n'.join('- '+description for description in dict.fromkeys(descriptions))
            if requirements:details+='\n- '+('Enabled only with: 'if english else'কার্যকর হওয়ার শর্ত: ')+', '.join(requirements)
            refs={}
            for source in evidence:
                if source.get('setting_id')in declared:
                    refs.setdefault(source['title'],set()).add(source['page'])
            if refs:details+='\n- '+('Source: 'if english else'সূত্র: ')+'; '.join(title+', PDF '+('pages 'if english else'পৃষ্ঠা ')+', '.join(map(str,sorted(pages)))for title,pages in refs.items())
            if assessment=='ruled_out':
                for key,values in constraints.items():
                    if key in known and known[key]not in values:
                        details+='\n'+('User-reported current value: 'if english else'আপনার দেওয়া বর্তমান মান: ')+setting_label(key)+' = '+known[key]
                for key,values in claim_exclusions.items():
                    if known.get(key)in values:
                        details+='\n'+('The manual excludes this branch when 'if english else'ম্যানুয়ালের ব্যতিক্রম অনুযায়ী এই শাখা প্রযোজ্য নয় যখন ')+setting_label(key)+' = '+known[key]
            numbers=', '.join(str(i+1)for i in sorted(intents))
            heading=('Symptom 'if english else'উপসর্গ ')+numbers+' — '+label
            rendered.append('**'+heading+'**\n\n'+details)
        else:
            rendered.append(claim['text'].strip())
    if covered!=set(range(len(queries))):
        raise ValueError('Diagnostic claims omitted a requested intent')
    if mechanical:
        groups = {}
        for claim in claims:
            groups.setdefault(tuple(claim['intent_ids']), []).append(claim['text'].strip())
        rendered = [('**' + ('Symptom ' if english else 'উপসর্গ ') +
                     ', '.join(str(i + 1) for i in intents) + '**\n\n' +
                     '\n'.join('- ' + text for text in texts)) for intents, texts in groups.items()]
    if re.search(r'\bfunction\b|ফাংশন',question,re.I):
        for intent in covered:
            owners={s['setting_id']for s in sources if (s.get('setting_id')or'').startswith('function:') and
                    intent in s.get('primary_intents',s.get('retrieval_intents',[])if not s.get('dependency')else[])}
            if len(owners)==1 and not owners<=declared_by_intent[intent]:
                raise ValueError('Diagnostic claim omitted the requested function definition')
    conflicts={key:values for key,values in assignments.items()
               if len(values)>1 or values.intersection(exclusions.get(key,set()))}
    if all(claim.get('assessment')=='ruled_out'for claim in claims):
        diagnosis='insufficient'
    if active_covered!=covered and diagnosis!='incompatible':
        diagnosis='insufficient'
    if conflicts and diagnosis!='incompatible':
        raise ValueError('Diagnostic claims have conflicting modes; report incompatible conditions')
    if diagnosis=='incompatible':
        prefix=('These explanations require separate possible modes; they do not establish one configuration and are not instructions to apply the modes together.' if english else
                'প্রতিটি উপসর্গের জন্য ম্যানুয়ালে প্রাসঙ্গিক সেটিং পাওয়া গেছে। তবে দুটি ব্যাখ্যার শর্ত পরস্পরবিরোধী, তাই একটি যৌথ কারণ নিশ্চিত করা যাচ্ছে না। নিচে আলাদা সম্ভাব্য ব্যাখ্যা ও শর্ত দেওয়া হলো।')
        explanations=[]
        for owner,values in conflicts.items():
            if len(values)>1:
                explanations.append(setting_label(owner)+(' requires different states across these explanations: 'if english else'-এর জন্য এই ব্যাখ্যাগুলোতে ভিন্ন অবস্থা প্রয়োজন: ')+', '.join(sorted(values))+'.')
        suffix=(('**Why the explanations conflict**'if english else'**দুই ব্যাখ্যা একসঙ্গে কেন মিলছে না**')+'\n\n'+' '.join(explanations)+(' A switch cannot have both states at the same time. Verify the actual settings and event sequence.'if english else' একই সময়ে একটি সুইচ দুটো অবস্থায় থাকতে পারে না। তাই কোন শর্তে কোন উপসর্গ ঘটছে এবং বর্তমান সেটিং যাচাই করা দরকার।'))if explanations else''
        if len(queries)==1:
            prefix='More than one separate setting mode can explain this observation; verify which is active.'if english else'একের বেশি আলাদা সেটিং অবস্থা এই পর্যবেক্ষণটি ব্যাখ্যা করতে পারে; কোনটি বর্তমানে আছে তা যাচাই করুন।'
    elif diagnosis=='insufficient':
        prefix='The excerpts do not establish a single cause for all symptoms.' if english else 'উদ্ধৃত তথ্য দিয়ে সব উপসর্গের একক কারণ নিশ্চিত করা যায়নি।'
        suffix=('Report which checks found a problem and when each symptom occurs.' if english else
                'কোন চেকে সমস্যা পাওয়া গেছে এবং কখন প্রতিটি উপসর্গ হচ্ছে তা জানান।') if mechanical else (
                'Clarify the settings and observations.' if english else 'সঠিক সেটিং ও পর্যবেক্ষণ জানিয়ে প্রশ্নটি পরিষ্কার করুন।')
    else:
        prefix=('These candidate conditions match the settings you reported; the machine itself has not been inspected.'if english else'এই সম্ভাব্য ব্যাখ্যার শর্তগুলো আপনার দেওয়া মানের সঙ্গে মেলে; মেশিনটি সরাসরি পরীক্ষা করা হয়নি।')if known else(
            'Possible manual-supported explanations; the current settings have not been inspected.' if english else 'ম্যানুয়াল অনুযায়ী সম্ভাব্য ব্যাখ্যা; মেশিনের বর্তমান সেটিং সরাসরি পরীক্ষা করা হয়নি।')
        suffix=''
    checks='' if mechanical else verification_questions(sources,english,owners={o for owners in declared_by_intent.values()for o in owners})
    if diagnosis=='insufficient'and known and not checks:
        checks=('With these current settings, does the symptom occur before the triggering action, after it, or both?'if english else
                'এই বর্তমান মানগুলো থাকা অবস্থায় সংশ্লিষ্ট ক্রিয়ার আগে, পরে, নাকি দুই অবস্থাতেই উপসর্গটি দেখা যায়?')
    if checks:checks=('**What to verify next**'if english else'**এখন যা যাচাই করবেন**')+'\n\n'+checks
    return '\n\n'.join(part for part in [prefix,*rendered,suffix,checks]if part).strip(),diagnosis
