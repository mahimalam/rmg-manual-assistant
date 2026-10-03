"""Question flow and durable, de-identified study records."""

import json
import uuid
import time
import re
import logging
from datetime import datetime

from app.db import utc_now
from app.language import model_ids, normalize_question, requested_english
from app.grounding import source_clarification
from app.diagnosis import source_rule_answer
from app.provider import ProviderError


logger = logging.getLogger(__name__)


def _manual_models(db, manual):
    text = manual['title'] + '\n' + manual['filename']
    with db.connect() as connection:
        cover = connection.execute('SELECT text FROM chunks WHERE manual_id=? AND page=1', (manual['id'],)).fetchall()
        tables = connection.execute("SELECT payload_json FROM knowledge_units WHERE manual_id=? AND page<=8 AND kind='table'",
                                    (manual['id'],)).fetchall()
    text += '\n' + '\n'.join(row['text'] for row in cover)
    for row in tables:
        text += '\n' + ' '.join(json.loads(row['payload_json'])['table_headers'])
    return model_ids(text)


def ask(db, retriever, provider, question, manual_ids=None, trial_id=None, page_hint=None,
        runtime=None, diagnostic_context=None, input_context=None):
    if not question.strip() or len(question.strip())>2000:
        raise ValueError('Enter a question of 1–2,000 characters')
    started=time.monotonic()
    usage_start=len(getattr(provider,'usage_log',[]))
    raw_start=len(getattr(provider,'raw_responses',[]))
    try:
        result = _ask(db,retriever,provider,question,manual_ids,trial_id,page_hint,runtime,diagnostic_context)
    except (RuntimeError,OSError,ValueError) as exc:
        query_id=uuid.uuid4().hex
        if isinstance(exc, (RuntimeError, OSError)):
            logger.exception("Question pipeline failed (%s)", query_id)
        error=str(exc)[:1000]
        evidence=getattr(exc,'evidence',[])
        search_question=getattr(exc,'search_question',None)
        status='provider_error' if isinstance(exc,ProviderError) else 'pipeline_error'
        answer='উত্তর তৈরি করা যায়নি। সংরক্ষিত ত্রুটির বিস্তারিত দেখুন; ম্যানুয়াল বা সংযোগ পরীক্ষা করে আবার চেষ্টা করুন।'
        timings={'total_seconds':round(time.monotonic()-started,3)}
        with db.connect()as connection:
            connection.execute('INSERT INTO queries (id,created_at,question,search_question,answer,status,sources_json,trial_id,error,provider_responses_json,provider_usage_json,timings_json,retrieval_release,translation_issue,evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (query_id,utc_now(),question.strip(),search_question,answer,status,'[]',trial_id,error,
                 json.dumps(getattr(provider,'raw_responses',[])[raw_start:],ensure_ascii=False),
                 json.dumps(getattr(provider,'usage_log',[])[usage_start:]),json.dumps(timings),
                 getattr(retriever,'release_id',None),getattr(provider,'translation_issue',None),json.dumps(evidence,ensure_ascii=False)))
        result = {'id':query_id,'answer':answer,'sources':[],'status':status,
                'retrieved_sources':evidence,'search_question':search_question,'timings':timings,'error':error,
                'translation_issue':getattr(provider,'translation_issue',None),
                'review_status':'Not independently engineering-validated'}
    if input_context is not None:
        with db.connect() as connection:
            connection.execute('UPDATE queries SET input_context_json=? WHERE id=?',
                               (json.dumps(input_context, ensure_ascii=False), result['id']))
        result['input_context'] = input_context
    return result


def _ask(db, retriever, provider, question, manual_ids=None, trial_id=None, page_hint=None,
        runtime=None, diagnostic_context=None):
    question = question.strip()
    if not question:
        raise ValueError("Enter or record a question")
    if len(question) > 2000:
        raise ValueError("Question exceeds 2,000 characters")
    started = time.monotonic()
    usage_start = len(getattr(provider, "usage_log", []))
    raw_start = len(getattr(provider, 'raw_responses', []))
    if page_hint:
        manual_id,page_number=page_hint
        if runtime is None:
            raise ValueError('Page rendering requires the runtime directory')
        if manual_ids is not None and manual_id not in manual_ids:
            raise ValueError('The selected page is outside the selected manuals')
        selected=next((m for m in db.manuals()if m['id']==manual_id and m['status']=='ready'),None)
        if not selected or type(page_number)is not int or not 1<=page_number<=selected['pages']:
            raise ValueError('Selected manual page does not exist')
        manual_ids=[manual_id]
    available = [manual for manual in db.manuals() if manual["status"] == "ready" and
                 (manual_ids is None or manual["id"] in manual_ids)]
    requested = model_ids(question)
    explicit, covered = [], set()
    for manual in available:
        models = _manual_models(db, manual)
        matches = [model for model in requested if any(model == alias or
                   (model.startswith(alias) and re.fullmatch(r'\d{2,4}[A-Z]?', model[len(alias):]))
                   for alias in models)]
        if matches:
            explicit.append(manual["id"])
            covered.update(matches)
    clarification = None
    operational = bool(re.search(r'\b(set|settings?|adjust|replace|remove|install|thread|needle|tension|oil|error|presser|motor)\b|সেটিং|ফাংশন|ত্রুটি|প্রেসার|বদল|খুল',
                                 normalize_question(question), re.I))
    operational = operational or bool(re.search(r'সুতা|ছিঁ[ড়ড়]|ছিড়|আওয়াজ|আওয়াজ|সাউন্ড|মেশিন.*সমস্যা', question))
    if available and requested and set(requested) - covered:
        clarification = 'প্রশ্নে উল্লেখ করা মেশিনের মডেলটি নির্বাচিত ম্যানুয়ালে নিশ্চিত করা যায়নি। সঠিক মডেলের ম্যানুয়াল আপলোড বা নির্বাচন করুন।'
    elif len(available) > 1 and not explicit and operational and not page_hint:
        clarification = 'কোন মেশিনের ম্যানুয়াল অনুযায়ী উত্তর চান? ম্যানুয়াল নির্বাচন করুন বা মডেলটি লিখুন।'
    if explicit:
        manual_ids = explicit
    search_question = (normalize_question(question)if clarification or not available else
                       (provider.search_query(question)if diagnostic_context.get('free_observations')else
                        diagnostic_context.get('search_question')or provider.search_query(diagnostic_context['root_question']))if diagnostic_context else provider.search_query(question))
    translated_at = time.monotonic()
    matches = [] if clarification or not available else retriever.search(search_question, manual_ids=manual_ids)
    if matches:
        if not page_hint and any(s.get('retrieval_queries')for s in matches):
            if diagnostic_context is None:
                ids={s['manual_id']for s in matches}
                diagnostic_context={'root_question':question,'observations':[],'known_settings':{},
                    'search_question':search_question,
                    'manuals':[{key:m[key]for key in ('id','sha256','knowledge_release')}for m in available if m['id']in ids],
                    'parent_query_id':None}
            for source in matches:
                source['known_settings']=diagnostic_context['known_settings']
                source['unstructured_observations']=diagnostic_context.get('free_observations',False)
        queries=next((source['retrieval_queries']for source in matches if source.get('retrieval_queries')),[])
        covered={intent for source in matches for intent in source.get('primary_intents',[])}
        if queries and covered!=set(range(len(queries))):
            clarification='প্রশ্নের সব উপসর্গের জন্য প্রাসঙ্গিক তথ্য পাওয়া যায়নি। অনুপস্থিত উপসর্গটি আলাদা করে জিজ্ঞাসা করুন বা সঠিক ম্যানুয়াল/পৃষ্ঠাটি দিন।'
        selected_ids = set(explicit or [manual['id'] for manual in available])
        if any(source['manual_id'] not in selected_ids for source in matches):
            clarification = 'প্রাপ্ত তথ্য নির্বাচিত ম্যানুয়ালের সঙ্গে মেলেনি। ম্যানুয়াল নির্বাচন করে সূচি পুনর্নির্মাণ করুন।'
        elif not page_hint and not explicit and len(available) > 1 and re.search(
                r'\b(adjust|setting|replace|thread|needle|tension|oil|error|motor|presser|noise|damaged)\b', search_question, re.I):
            clarification = 'কোন মেশিনের ম্যানুয়াল অনুযায়ী উত্তর চান? ম্যানুয়াল নির্বাচন করুন বা মডেলটি লিখুন।'
    retrieved_at = time.monotonic()
    if page_hint:
        if runtime is None:
            raise ValueError("Page rendering requires the runtime directory")
        manual_id, page_number = page_hint
        if manual_ids and manual_id not in manual_ids:
            raise ValueError("The selected page is outside the selected manuals")
        with db.connect() as connection:
            manual = connection.execute("SELECT * FROM manuals WHERE id=? AND status='ready'",
                                        (manual_id,)).fetchone()
        if not manual or page_number < 1 or page_number > manual["pages"]:
            raise ValueError("Selected manual page does not exist")
        from app.pages import render_page

        page_rows = [row for row in db.chunks([manual_id]) if row["page"] == page_number]
        page_text = "\n".join(row["text"] for row in page_rows)
        if len(page_text) > 30000:
            raise ValueError("Selected page evidence is too large; narrow the question")
        matches.insert(0, {"id": f"{manual_id}:{page_number}:image", "manual_id": manual_id,
                           "title": manual["title"], "page": page_number,
                           "score": 0.0, "text": page_text
                           or "[Selected page image; no OCR text]",
                           "page_image": render_page(runtime, manual_id, page_number)})
    validation_error=None
    reasoning_path='model'
    if clarification:
        answer, used, status = clarification, [], 'clarify'
    elif not matches:
        answer, used, status = "এই ম্যানুয়ালে অনুসন্ধানযোগ্য তথ্য পাওয়া যায়নি।", [], "no_evidence"
    else:
        ruled=source_rule_answer(matches,requested_english(question),question)
        if ruled:
            answer,ids,rule_status=ruled
            provider.last_answer_status=rule_status
            reasoning_path='source_rules'
        else:
            try:
                answer, ids = provider.answer(question, matches)
            except ProviderError as exc:
                if getattr(provider,'validation_issue',None) and any(s.get('retrieval_queries')for s in matches):
                    validation_error=provider.validation_issue
                    answer,ids=source_clarification(matches,requested_english(question))
                    provider.last_answer_status='clarify'
                    reasoning_path='source_clarification'
                else:
                    exc.evidence=[{key:value for key,value in source.items()if key!='page_image'}for source in matches]
                    exc.search_question=search_question
                    raise
        used_by_page = {}
        for i in ids:
            source = matches[i - 1]
            key = (source["manual_id"], source["page"])
            if key not in used_by_page:
                used_by_page[key] = dict(source)
            else:
                existing = used_by_page[key]
                if source["text"] not in existing["text"]:
                    existing["text"] += "\n\n" + source["text"]
                existing["score"] = max(existing["score"], source["score"])
                if source.get("page_image"):
                    existing["page_image"] = source["page_image"]
                existing["flags"] = list(dict.fromkeys(existing.get("flags", []) + source.get("flags", [])))
        used = list(used_by_page.values())
        status = getattr(provider, "last_answer_status", "answered") if used else "refused"
    query_id = uuid.uuid4().hex
    timings = {"translation_seconds": round(translated_at - started, 3),
               "retrieval_seconds": round(retrieved_at - translated_at, 3),
               "answer_seconds": round(time.monotonic() - retrieved_at, 3),
               "total_seconds": round(time.monotonic() - started, 3)}
    evidence = [{key: value for key, value in source.items() if key != "page_image"}
                for source in matches]
    if diagnostic_context:diagnostic_context['reasoning_path']=reasoning_path
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO queries (id, created_at, question, search_question, answer, status, "
            "sources_json, trial_id,evidence_json,provider_usage_json,timings_json,retrieval_release,provider_responses_json,translation_issue,error,diagnostic_context_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query_id, utc_now(), question, search_question, answer, status,
             json.dumps([{key: source[key] for key in ("id", "title", "page", "score", "text")}
                         for source in used], ensure_ascii=False), trial_id,
             json.dumps(evidence, ensure_ascii=False), json.dumps(getattr(provider, "usage_log", [])[usage_start:]),
             json.dumps(timings), getattr(retriever, "release_id", None),
             json.dumps(getattr(provider,'raw_responses',[])[raw_start:],ensure_ascii=False),
             getattr(provider,'translation_issue',None),validation_error,
             json.dumps(diagnostic_context,ensure_ascii=False)if diagnostic_context else None),
        )
    return {"id": query_id, "answer": answer, "sources": used, "status": status,
            "retrieved_sources": evidence,
            "search_question": search_question, "timings": timings,
            "translation_issue": getattr(provider, 'translation_issue', None),
            "error":validation_error,
            "diagnostic_context":diagnostic_context,
            "reasoning_path":reasoning_path,
            "review_status": "Not independently engineering-validated"}


def continue_diagnosis(db,retriever,provider,query_id,reply,trial_id=None):
    if not reply.strip()or len(reply)>500:
        raise ValueError('Provide current observations in 1–500 characters')
    with db.connect()as c:
        parent=c.execute('SELECT diagnostic_context_json FROM queries WHERE id=?',(query_id,)).fetchone()
    if not parent or not parent['diagnostic_context_json']:
        raise ValueError('This question has no diagnostic case; ask the full question again')
    context=json.loads(parent['diagnostic_context_json'])
    available={m['id']:m for m in db.manuals()if m['status']=='ready'}
    for snapshot in context['manuals']:
        if snapshot['id']not in available or any(available[snapshot['id']][key]!=snapshot[key]for key in ('sha256','knowledge_release')):
            raise ValueError('The diagnostic manual changed or was updated; ask the full question again')
    text=normalize_question(reply)
    observations={}
    remainder=text
    for label,kind in ((r'(?:Function(?:\s+(?:No|number)\.?)?|ফাংশন(?:\s+(?:নম্বর|নং))?)','function'),
                       (r'(?:DIP\s+switch|ডিপ\s+সুইচ)','dip')):
        for match in re.finditer(label+r'\s*(\d+)\s*[=:]\s*(-?\d+(?:\.\d+)?|ON|OFF|অন|অফ)\b',text,re.I):
            owner=kind+':'+match[1];value={'অন':'ON','অফ':'OFF'}.get(match[2].upper(),match[2].upper())
            if owner in observations and observations[owner]!=value:
                raise ValueError('Conflicting current values in this observation; clarify the setting')
            observations[owner]=value
            remainder=remainder.replace(match[0],'')
    if re.search(r'[A-Za-z\u0980-\u09ff]',remainder):context['free_observations']=True
    context['known_settings'].update(observations)
    context['observations'].append(text)
    context['parent_query_id']=query_id
    question=context['root_question']+'\n\nUser-reported observations (chronological; later explicit corrections supersede earlier values):\n'+'\n'.join(context['observations'])
    if len(question)>2000:
        raise ValueError('The diagnostic history is too long; ask a new concise question with the current observations')
    return ask(db,retriever,provider,question,[m['id']for m in context['manuals']],trial_id,
               diagnostic_context=context)


def save_feedback(db, query_id, verdict, note):
    if verdict not in ("correct", "incorrect", "unclear"):
        raise ValueError("Unknown feedback verdict")
    with db.connect() as connection:
        connection.execute("INSERT INTO feedback VALUES (?,?,?,?) ON CONFLICT(query_id) "
                           "DO UPDATE SET verdict=excluded.verdict,note=excluded.note,"
                           "created_at=excluded.created_at", (query_id, verdict, note[:4000], utc_now()))


def export_feedback(db):
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["query_id", "created_at", "question", "search_question", "answer", "status",
                     "verdict", "note", "timings_json", "retrieval_release", "sources_json",
                     "evidence_json", "provider_usage_json", "error", "provider_responses_json", "translation_issue", "diagnostic_context_json", "input_context_json", "review_basis"])
    with db.connect() as connection:
        rows = connection.execute("SELECT q.*,f.verdict,f.note FROM queries q "
                                  "LEFT JOIN feedback f ON f.query_id=q.id ORDER BY q.created_at").fetchall()
    for row in rows:
        values = [row["id"], row["created_at"], row["question"], row["search_question"], row["answer"], row["status"],
                         row["verdict"], row["note"], row["timings_json"], row["retrieval_release"],
                  row["sources_json"], row["evidence_json"], row["provider_usage_json"], row['error'],
                  row['provider_responses_json'],row['translation_issue'],row['diagnostic_context_json'], row['input_context_json'], "personal testing; not domain approval"]
        writer.writerow(["'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                         else value for value in values])
    return output.getvalue().encode("utf-8-sig")


def start_trial(db, condition: str, scenario: str):
    if condition not in ("assistant", "pdf"):
        raise ValueError("Unknown comparison condition")
    if not scenario.strip():
        raise ValueError("Enter the scenario/question shown to the participant")
    trial_id = uuid.uuid4().hex
    with db.connect() as connection:
        if connection.execute("SELECT 1 FROM trials WHERE status='active' LIMIT 1").fetchone():
            raise ValueError("Finish or interrupt the active trial first")
        connection.execute(
            "INSERT INTO trials (id, created_at, condition, scenario, status) "
            "VALUES (?, ?, ?, ?, 'active')",
            (trial_id, utc_now(), condition, scenario.strip()),
        )
    return trial_id


def interrupt_trial(db, trial_id: str):
    with db.connect() as connection:
        connection.execute(
            "UPDATE trials SET status='interrupted', ended_at=? WHERE id=? AND status='active'",
            (utc_now(), trial_id),
        )


def finish_trial(db, trial_id: str, response: str):
    if not response.strip():
        raise ValueError("Enter the participant's final response")
    with db.connect() as connection:
        trial = connection.execute("SELECT * FROM trials WHERE id=?", (trial_id,)).fetchone()
        if trial is None:
            raise ValueError("Trial not found")
        if trial["status"] == "complete":
            return dict(trial)
        if trial["status"] != "active":
            raise ValueError("Trial is not active")
        elapsed = max(0.0, (datetime.fromisoformat(utc_now()) -
                            datetime.fromisoformat(trial["created_at"])).total_seconds())
        connection.execute(
            "UPDATE trials SET status='complete', final_answer=?, ended_at=?, "
            "elapsed_seconds=? WHERE id=? AND status='active'",
            (response.strip(), utc_now(), elapsed, trial_id),
        )
        return dict(connection.execute("SELECT * FROM trials WHERE id=?", (trial_id,)).fetchone())


def export_study(db) -> bytes:
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["trial_id", "condition", "scenario", "status", "started_at_utc",
                     "ended_at_utc", "elapsed_seconds", "final_answer", "query_count"])
    with db.connect() as connection:
        rows = connection.execute("""
            SELECT t.*, COUNT(q.id) AS query_count FROM trials t
            LEFT JOIN queries q ON q.trial_id=t.id GROUP BY t.id ORDER BY t.created_at
        """).fetchall()
    for row in rows:
        writer.writerow([row["id"], row["condition"], row["scenario"], row["status"],
                         row["created_at"], row["ended_at"], row["elapsed_seconds"],
                         row["final_answer"], row["query_count"]])
    return output.getvalue().encode("utf-8-sig")


def export_queries(db) -> bytes:
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["query_id", "trial_id", "created_at_utc", "question", "search_question",
                     "status", "answer",
                     "source_title", "source_page", "source_score", "source_text", "input_context_json"])
    with db.connect() as connection:
        rows = connection.execute("SELECT * FROM queries ORDER BY created_at").fetchall()
    for row in rows:
        sources = json.loads(row["sources_json"]) or [None]
        for source in sources:
            writer.writerow([row["id"], row["trial_id"], row["created_at"], row["question"],
                             row["search_question"], row["status"], row["answer"],
                             source["title"] if source else "",
                             source["page"] if source else "",
                             source["score"] if source else "",
                             source["text"] if source else "", row['input_context_json']])
    return output.getvalue().encode("utf-8-sig")
