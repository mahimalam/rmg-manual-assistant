"""Real image-only excerpt upload, automatic enrichment, restart and removal."""

import json
import re
from pathlib import Path

import pymupdf

from app.config import Settings
from app.db import Database, utc_now
from app.ingest import add_pdf, delete_pdf
from app.language import normalize_question
from app.retrieval import Retriever
from app.service import ask
from app.structured import BudgetPolicy, atomic_json, enrich_recommended, process_manual
from evaluation.budget import BudgetedProvider, RunBudget


def run():
    s=Settings.load()
    runtime=s.runtime/'upload-lifecycle-development'
    if (runtime/'study.sqlite3').exists():
        raise ValueError('This one-time real upload experiment already exists; inspect its report rather than rebilling')
    original=s.root/'evaluation/manuals/Juki-DDL9000CS-instruction.pdf'
    with pymupdf.open(original)as source:
        image=source[7].get_pixmap(dpi=180,alpha=False).tobytes('png')
        doc=pymupdf.open()
        doc.set_metadata({'title':'DDL-9000C-SMS development scanned excerpt'})
        doc.new_page().insert_text((50,50),'DDL-9000C-SMS development scanned excerpt')
        page=doc.new_page(width=source[7].rect.width,height=source[7].rect.height)
        page.insert_image(page.rect,stream=image)
        pdf=doc.tobytes();doc.close()
    db=Database(runtime/'study.sqlite3')
    report={'created_at':utc_now(),'source':str(original),'original_pdf_page':8,
            'interpretation':'Image-only derivative of manufacturer source; development validation, not independent review.'}
    provider=BudgetedProvider(s,RunBudget(s))
    m,added=add_pdf(db,runtime,'scanned-excerpt.pdf',pdf,publish_baseline=False)
    r=Retriever(db,runtime,s.embedding_model)
    base=process_manual(db,runtime,m['id'],retriever=r)
    enrichment=enrich_recommended(db,runtime,m['id'],provider,BudgetPolicy(3,10,50,2),retriever=r)
    duplicate,added_again=add_pdf(db,runtime,'duplicate.pdf',pdf)
    repeated=enrich_recommended(db,runtime,m['id'],provider,BudgetPolicy(3,10,50,2),retriever=r)
    reopened=Database(runtime/'study.sqlite3')
    rr=Retriever(reopened,runtime,s.embedding_model,'cross-encoder/ms-marco-MiniLM-L6-v2')
    result=ask(reopened,rr,provider,'For DDL-9000C-SMS, distinguish the standard support rod height from the value with AK.',[m['id']])
    answer=normalize_question(result['answer'])
    report.update(baseline=base,enrichment=enrichment,repeated_enrichment=repeated,
                  duplicate_deduplicated=not added_again and duplicate['id']==m['id'],
                  answer=result,answer_numeric_check=all(re.search(n,answer) for n in ('63','68','33','38')),
                  usage=provider.usage_log)
    atomic_json(s.runtime/'evaluations/adversarial_new_upload.json',report)
    delete_pdf(reopened,runtime,m['id']);rr.remove_manual(m['id'])
    report['removal_check']=not reopened.manuals() and not reopened.chunks() and rr.collection.count()==0
    report['paid_image_calls']=sum(a['finish_reason'] is not None for a in provider.usage_log[:1])
    atomic_json(s.runtime/'evaluations/adversarial_new_upload.json',report)
    print('AUTOMATIC',enrichment,'DUPLICATE',report['duplicate_deduplicated'],'ANSWER',report['answer_numeric_check'],'REMOVAL',report['removal_check'],flush=True)


if __name__=='__main__':run()
