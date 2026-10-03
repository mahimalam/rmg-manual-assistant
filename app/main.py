"""Screen implementation; launch the project-root main.py with Streamlit."""

import json

import streamlit as st

from app.config import Settings
from app.db import Database
from app.ingest import add_pdf, delete_pdf
from app.interface import update_recording, voice_submission
from app.language import contains_devanagari_letters
from app.pages import render_page
from app.provider import ManualProvider, ProviderError
from app.retrieval import Retriever
from app.service import (ask, export_queries, export_study, finish_trial, interrupt_trial,
                         start_trial, save_feedback, export_feedback, continue_diagnosis)
from app.structured import (BudgetPolicy, atomic_json, process_manual, replace_manual,
                            save_visual_correction, enrich_recommended)
from app.voice import speak, manual_keyterms


st.set_page_config(page_title="RMG Manual Assistant", page_icon="📖", layout="wide")
settings = Settings.load()
db = Database(settings.runtime / "study.sqlite3")


@st.cache_resource
def get_retriever(path, model, reranker_model=None):
    return Retriever(db, path, model, reranker_model)


st.html((settings.root / "app" / "styles.css").read_text())
st.caption("ENGINEERING KNOWLEDGE WORKSPACE")
st.title("RMG Manual Assistant")
st.caption("Ask in Bangla or English. Answers are linked to your saved manuals.")
library_snapshot = [manual for manual in db.manuals() if manual["status"] == "ready"]
library_metric, pages_metric, voice_metric = st.columns(3)
library_metric.metric("Available manuals", len(library_snapshot))
pages_metric.metric("Manual pages", sum(manual["pages"] for manual in library_snapshot))
voice_metric.metric("Voice recognition", "Cloud + local" if settings.deepgram_api_key else "Local")

questions, library, study = st.tabs(["Ask the manuals", "Manual library", "Study workspace"])

with library:
    st.subheader("Upload engineering manuals")
    st.write("New PDFs are processed into local paragraphs, procedures and table records. "
             "Visual enrichment sends selected page images to your configured AI provider once, "
             "then saves the results for reuse. PDF files and stored knowledge stay on this computer.")
    budget = None
    with st.expander("Visual extraction cost controls"):
        st.write("Local processing is free of API charges. These limits apply to image extraction; "
                 "ordinary answer calls are separate. Enter your gateway's verified rates, "
                 "not rates assumed from the model name.")
        budget_path = settings.runtime / "ingestion-settings.json"
        saved = json.loads(budget_path.read_text()) if budget_path.exists() else {}
        ceiling = st.number_input("Total image-extraction allowance (USD)", min_value=0.0,
                                  value=float(saved.get("limit_usd", 0)), step=1.0)
        input_rate = st.number_input("Gateway input price per million tokens (USD)", min_value=0.0,
                                     value=float(saved.get("input_rate", 0)), step=0.1)
        output_rate = st.number_input("Gateway output price per million tokens (USD)", min_value=0.0,
                                      value=float(saved.get("output_rate", 0)), step=0.1)
        max_calls = st.number_input("Maximum image requests in this library", min_value=0,
                                    value=int(saved.get("max_calls", 0)), step=1)
        verified = st.checkbox("I have verified these gateway prices", value=bool(saved.get('verified',False)))
        automatic = st.checkbox("Automatically enrich difficult pages after upload", value=bool(saved.get('automatic',False)),
                                help="Prioritizes scans and ambiguous tables, then diagrams. Stops at the saved budget. Failed calls require explicit review.")
        spend = db.ingestion_spend()
        st.caption(f"{spend['attempts']} attempts; ${spend['accounted_usd']:.4f} accounted from "
                   "reported tokens or conservative reservations. This is not a billing receipt.")
        if verified and ceiling > 0 and input_rate > 0 and output_rate > 0 and max_calls > 0:
            budget = BudgetPolicy(ceiling, input_rate, output_rate, int(max_calls))
        if st.button('Save visual extraction settings'):
            atomic_json(budget_path, {'limit_usd':ceiling,'input_rate':input_rate,'output_rate':output_rate,
                                     'max_calls':int(max_calls),'verified':verified,'automatic':automatic})
            st.success('Visual extraction settings saved.')
    files = st.file_uploader("PDF manuals", type="pdf", accept_multiple_files=True)
    if st.button("Add selected PDFs", disabled=not files):
        for file in files or []:
            try:
                with st.spinner(f"Extracting {file.name}; scanned pages may take a while"):
                    manual, added = add_pdf(db, settings.runtime, file.name, file.getvalue(),
                                            publish_baseline=False)
                    progress = st.progress(0, text="Building structured knowledge")
                    process_manual(db, settings.runtime, manual["id"], progress=lambda done, total:
                                   progress.progress(done / total, text=f"Processed {done}/{total} pages"),
                                   retriever=get_retriever(settings.runtime, settings.embedding_model))
                if automatic and budget is not None:
                    with st.spinner("Enriching difficult pages within the saved allowance"):
                        enrichment = enrich_recommended(db, settings.runtime, manual["id"],
                            ManualProvider(settings.base_url, settings.api_key, settings.model),budget,
                            retriever=get_retriever(settings.runtime, settings.embedding_model))
                    if enrichment['error']:
                        st.warning("Local knowledge is available. Visual extraction needs review: " + enrichment['error'])
                    if enrichment['deferred_pages']:
                        st.caption(f"{enrichment['deferred_pages']} visual candidates deferred by the allowance; inspect the extraction report.")
                st.success(f"{manual['title']}: {'added' if added else 'already in library'}")
            except ValueError as exc:
                st.error(f"{file.name}: {exc}")
        try:
            with st.spinner("Building the local search index. First run downloads the embedding model."):
                vector_progress = st.progress(0, text="Indexing knowledge units")
                get_retriever(settings.runtime, settings.embedding_model).sync(progress=lambda done, total:
                    vector_progress.progress(done / total, text=f"Indexed {done}/{total} units"))
        except Exception as exc:
            st.error(f"Indexing failed: {exc}. The PDF is stored; use Rebuild index to retry.")
    all_manuals = db.manuals()
    if all_manuals:
        if st.button("Build/update local knowledge for the library"):
            try:
                progress = st.progress(0)
                for manual in all_manuals:
                    if manual["status"] != "retired":
                        process_manual(db, settings.runtime, manual["id"],
                                       progress=lambda done, total: progress.progress(done / total),
                                       retriever=get_retriever(settings.runtime, settings.embedding_model))
                get_retriever(settings.runtime, settings.embedding_model).sync(progress=lambda done, total:
                    progress.progress(done / total, text=f"Indexed {done}/{total} units"))
                st.success("Structured knowledge and search index are ready.")
                st.rerun()
            except Exception as exc:
                st.error(f"Knowledge build stopped: {exc}. Completed work is saved for resume.")
    manuals = [manual for manual in db.manuals() if manual["status"] == "ready"]
    if all_manuals:
        for manual in all_manuals:
            col1, col2 = st.columns([5, 1])
            col1.write(f"**{manual['title']}** — {manual['pages']} pages — "
                       f"{manual['status']} / {manual['knowledge_status']}")
            if col2.button("Remove", key=f"remove-{manual['id']}"):
                delete_pdf(db, settings.runtime, manual["id"])
                try:
                    get_retriever(settings.runtime, settings.embedding_model).remove_manual(
                        manual["id"])
                except Exception:
                    st.warning("Manual removed; rebuild the search index to clear derived vectors.")
                st.session_state.pop("result", None)
                st.rerun()
            with st.expander(f"Inspect or improve {manual['title']}"):
                job = db.job(manual["id"])
                if job:
                    st.caption(f"Processing: {job['status']}; {job['pages_done']}/{manual['pages']} pages")
                    if job["error"]:
                        st.warning(job["error"])
                units = db.units(manual["id"])
                st.write(f"{len(units)} knowledge units. Engineering approval: not established.")
                page = st.number_input("Inspect PDF page", min_value=1, max_value=manual["pages"],
                                        value=1, key=f"inspect-{manual['id']}")
                if st.checkbox("Show extracted page", key=f"show-inspect-{manual['id']}"):
                    st.image(render_page(settings.runtime, manual["id"], int(page)))
                    page_units = [json.loads(unit["payload_json"]) for unit in units if unit["page"] == page]
                    for unit in page_units:
                        st.write(f"**{unit['kind']}: {unit['title']}**")
                        st.text(unit["text"])
                        if unit["kind"] == "table":
                            st.json({"headers": unit["table_headers"], "rows": unit["table_rows"],
                                     "footnotes": unit["footnotes"]}, expanded=False)
                        if unit["flags"] or unit["blocking_issues"]:
                            st.caption("Review: " + "; ".join(unit["flags"] + unit["blocking_issues"]))
                if manual["knowledge_release"]:
                    release_folder = (settings.runtime / "knowledge" / manual["id"] /
                                      "releases" / manual["knowledge_release"])
                    manifest = json.loads((release_folder / "manifest.json").read_text())
                    recommended = manifest["pages_needing_visual_review"]
                    st.caption(f"{len(manifest['visual_pages'])} pages visually enriched; "
                               f"{len(recommended)} pages still need visual inspection.")
                    st.download_button("Download extraction report", json.dumps(manifest, ensure_ascii=False, indent=2),
                                       file_name=f"{manual['id']}-extraction-report.json",
                                       key=f"report-{manual['id']}")
                    st.download_button("Download knowledge units", (release_folder / "units.jsonl").read_bytes(),
                                       file_name=f"{manual['id']}-units.jsonl", key=f"units-{manual['id']}")
                else:
                    recommended = list(range(1, manual["pages"] + 1))
                visual_pages = st.multiselect("Pages to enrich with the VLM", list(range(1, manual["pages"] + 1)),
                                              key=f"visual-{manual['id']}")
                st.caption("Choose a small diverse pilot first. Successful pages are cached; failures "
                           "retain their raw output and do not automatically repeat a paid call.")
                retry_failed = st.checkbox("Allow another paid attempt for failed selected pages",
                                            key=f"retry-{manual['id']}",
                                            help="A previous timed-out request may already have been billed. "
                                            "Successful cached pages are still reused.")
                if st.button("Extract selected pages once", disabled=not visual_pages or budget is None
                             or manual["status"] == "retired", key=f"enrich-{manual['id']}"):
                    try:
                        atomic_json(budget_path, {"limit_usd": ceiling, "input_rate": input_rate,
                                                 "output_rate": output_rate, "max_calls": int(max_calls)})
                        progress = st.progress(0)
                        process_manual(db, settings.runtime, manual["id"],
                                       ManualProvider(settings.base_url, settings.api_key, settings.model),
                                       budget, visual_pages, progress=lambda done, total: progress.progress(done / total),
                                       retry_failed=retry_failed,
                                       retriever=get_retriever(settings.runtime, settings.embedding_model))
                        get_retriever(settings.runtime, settings.embedding_model).sync()
                        st.success("Visual results saved and indexed.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Visual enrichment stopped: {exc}. Saved successful outputs are reusable.")
                response_files = list((settings.runtime / "knowledge" / manual["id"] / "visual").glob("*.json"))
                originals = [json.loads(path.read_text()) for path in response_files]
                originals = [item for item in originals if item.get("source_context", {}).get("pdf_page") == page]
                initial = originals[-1]["content"] if originals else json.dumps({"units": []}, indent=2)
                correction = st.text_area("Optional corrected extraction JSON for this page", value=initial,
                                           key=f"correction-{manual['id']}-{page}", height=140)
                if st.button("Save my correction and reindex", key=f"correct-{manual['id']}",
                             disabled=manual["status"] == "retired"):
                    try:
                        save_visual_correction(db, settings.runtime, manual["id"], int(page), correction,
                                               retriever=get_retriever(settings.runtime, settings.embedding_model))
                        get_retriever(settings.runtime, settings.embedding_model).sync()
                        st.success("User correction saved; it does not establish independent domain approval.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Correction could not be saved: {exc}")
                replacement = st.file_uploader("Replace with a revised PDF", type="pdf",
                                                key=f"replacement-{manual['id']}")
                if st.button("Process replacement", disabled=replacement is None,
                             key=f"replace-{manual['id']}"):
                    try:
                        replace_manual(db, settings.runtime, manual["id"], replacement.name,
                                       replacement.getvalue(), get_retriever(settings.runtime, settings.embedding_model))
                        st.session_state.pop("result", None)
                        st.success("Replacement ready. The old revision is retained as retired until you remove it.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Replacement stopped: {exc}. The previous revision is retained.")
        if st.button("Rebuild index"):
            try:
                with st.spinner("Rebuilding local vectors"):
                    total = get_retriever(settings.runtime, settings.embedding_model).rebuild()
                st.success(f"Indexed {total} text passages")
            except Exception as exc:
                st.error(f"Index rebuild failed: {exc}")
    else:
        st.info("Upload a manual to begin.")

with questions:
    if not manuals:
        st.info("Add a PDF in the Manual library first.")
    else:
        names = {f"{m['title']} ({m['id'][:6]})": m["id"] for m in manuals}
        with st.expander("Manuals and search options"):
            selected = st.multiselect("Search these manuals", list(names), default=list(names),
                                      help="All saved manuals are selected by default. Narrow this for one machine.")
            use_reranker = st.checkbox("Use local evidence reranker", value=True,
                                       help="Ranks candidate passages locally; no extra AI-provider call. "
                                       "First use downloads a small English ranking model.")
            diagram_question = st.checkbox("Question about a diagram or a particular page",
                                           help="Sends that page image to the AI provider on each Ask, with separate charges. "
                                           "For reusable visual knowledge, use cached VLM extraction in Manual library.")
        st.caption(f"Searching {len(selected)} saved manual{'s' if len(selected) != 1 else ''}.")
        page_hint = None
        if diagram_question and selected:
            page_manual_name = st.selectbox("Manual containing that page", selected)
            page_manual_id = names[page_manual_name]
            page_count = next(m["pages"] for m in manuals if m["id"] == page_manual_id)
            page_number = st.number_input("PDF page number (counting from the first PDF page)",
                                          min_value=1, max_value=page_count, value=1)
            page_hint = (page_manual_id, int(page_number))
            with st.expander("Preview selected page"):
                st.image(render_page(settings.runtime, page_manual_id, int(page_number)))
        st.subheader("Ask your manuals")
        mode = st.radio("Input mode", ["Type", "Voice"], horizontal=True, key="input_mode")
        if mode == "Voice":
            st.caption("Record, stop, then Ask. You can edit the recognized question if needed.")
            language_label = st.selectbox("Recording language", ["Bangla", "English", "Auto detect"],
                                          key="recording_language",
                                          help="Bangla is the default. Choose English for an English recording; use Auto detect for other languages.")
            recording_language = {"Auto detect":"auto", "Bangla":"bn", "English":"en"}[language_label]
            audio = st.audio_input("Record your question", sample_rate=16000, key="recorded_question")
            keyterms = manual_keyterms(db, [names[name] for name in selected]) if audio is not None and settings.deepgram_api_key else []
            if audio is not None:
                with st.spinner("Recognizing and preparing your question…"):
                    update_recording(audio.getvalue(), st.session_state,
                                     settings.whisper_model, settings.deepgram_api_key, False,
                                     recording_language,
                                     ManualProvider(settings.base_url, settings.api_key, settings.model), keyterms)
            if audio is not None and (st.session_state.get("voice_error") or
                                      st.session_state.get("voice_preparation_error")):
                if st.button("Retry question preparation"):
                    with st.spinner("Retrying question preparation…"):
                        update_recording(audio.getvalue(), st.session_state, settings.whisper_model,
                                         settings.deepgram_api_key, True, recording_language,
                                         ManualProvider(settings.base_url, settings.api_key, settings.model), keyterms)
                    st.rerun()
            if st.session_state.get("voice_error"):
                st.error(f"Recognition failed: {st.session_state['voice_error']}")
            if st.session_state.get("voice_preparation_error"):
                st.error(st.session_state["voice_preparation_error"])
                st.caption("The original transcript is retained below. Bengali or English transcripts can still be submitted directly.")
            if st.session_state.get("voice_transcript"):
                with st.expander("Original transcript · check what was heard", expanded=True):
                    st.write(st.session_state["voice_transcript"])
                    st.caption("Missing or wrong words cannot be recovered by translation. Correct the question below or record again.")
            probability = st.session_state.get("voice_language_confidence")
            if probability is not None and probability < 0.6:
                st.warning("Language detection is uncertain. Check the transcript or select the recording language explicitly.")
            for warning in st.session_state.get("voice_warnings", []):
                st.warning(warning)
            if st.session_state.get("voice_source"):
                st.caption(f"Recognized with {st.session_state['voice_source']} · language: {st.session_state['voice_language']}. "
                           "Bengali preparation keeps the original question available for search.")
                if st.session_state.get('voice_source') == 'Local fallback':
                    st.warning(st.session_state.get('voice_recognition', {}).get('fallback_reason', 'Deepgram is unavailable.') +
                               ' This recording used local recognition. You can edit or Ask directly.')
            elif not settings.deepgram_api_key:
                st.caption("Local recognition is active. Allow microphone access; the first use can take a few minutes.")
            if not settings.deepgram_api_key:
                st.info('Deepgram is not configured. Local recognition is active until a valid key is added to the private configuration.')
        active_pdf_trial = False
        if st.session_state.get("trial_id"):
            with db.connect() as connection:
                active = connection.execute("SELECT condition, status FROM trials WHERE id=?",
                                            (st.session_state["trial_id"],)).fetchone()
            active_pdf_trial = bool(active and active["status"] == "active" and
                                    active["condition"] == "pdf")
        if active_pdf_trial:
            st.info("Assistant questions are disabled during a PDF comparison trial.")
        with st.form(f"question-form-{mode.lower()}"):
            question = st.text_area("Your question" if mode == "Type" else
                                    "Correct the question" if st.session_state.get("voice_preparation_error") else
                                    "Prepared Bengali question",
                                    key="question" if mode == "Type" else "voice_question",
                                    persist_state="session", height=130,
                                    placeholder="Include the machine model and describe what you need to know…")
            submitted = st.form_submit_button("Ask", type="primary", width="stretch",
                                              disabled=not selected or active_pdf_trial or (mode == "Voice" and
                                                       bool(st.session_state.get("voice_error"))))
        if submitted and not question.strip():
            st.warning("Enter a question or record one before asking.")
        elif submitted and contains_devanagari_letters(question):
            st.warning("Correct the question to Bengali or English before asking.")
        elif submitted:
            try:
                input_context = None
                if mode == "Voice":
                    question, input_context = voice_submission(question, st.session_state)
                with st.spinner("Searching manuals and preparing an answer"):
                    retriever = get_retriever(settings.runtime, settings.embedding_model,
                                             "cross-encoder/ms-marco-MiniLM-L6-v2" if use_reranker else None)
                    retriever.sync()
                    result = ask(db, retriever,
                                 ManualProvider(settings.base_url, settings.api_key, settings.model),
                                 question, [names[name] for name in selected],
                                 st.session_state.get("trial_id"), page_hint, settings.runtime,
                                 input_context=input_context)
                st.session_state["result"] = result
                st.session_state.pop("answer_audio", None)
            except (ValueError, ProviderError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Question failed: {exc}")
        result = st.session_state.get("result")
        if result:
            st.subheader("Answer")
            st.markdown(result["answer"])
            st.caption("Check the cited manual pages before changing machine settings.")
            if result.get('error'):
                if result['status']=='clarify':
                    st.caption('The generated diagnosis did not pass the source checks. Verify the manual excerpts below.')
                else:
                    st.error(result['error'])
            if st.button("Read answer aloud in Bangla"):
                try:
                    with st.spinner("Generating Bengali audio locally; long answers can take about a minute"):
                        st.session_state["answer_audio"] = speak(result["answer"])
                except Exception as exc:
                    st.error(f"Speech synthesis failed: {exc}")
            if st.session_state.get("answer_audio"):
                st.audio(st.session_state["answer_audio"], format="audio/wav")
            with st.expander("Sources and answer details"):
                st.caption(f"Status: {result['status']} • {result.get('timings', {}).get('total_seconds', 0):.2f}s "
                           "• Personal testing; not independent engineering validation")
                if result.get('reasoning_path') == 'source_rules':
                    st.caption('Explanation derived from the retrieved manual setting rules and your reported observations.')
                if result.get('translation_issue'):
                    st.caption('Search used the original question because its English interpretation lost details.')
                if result.get("search_question") and result["search_question"] != question:
                    st.caption(f"English search interpretation: {result['search_question']}")
                if result["sources"]:
                    st.write("Cited pages")
                    for source_index, source in enumerate(result["sources"]):
                        st.write(f"**{source['title']} — PDF page {source['page']}**")
                        if source.get("flags"):
                            st.warning("; ".join(source["flags"]))
                        st.text(source["text"])
                        if st.checkbox("Show page image", key=f"source-page-{result['id']}-{source_index}"):
                            st.image(render_page(settings.runtime, source["manual_id"], source["page"]))
                if result.get("retrieved_sources"):
                    if st.checkbox("Show all retrieved passages", key=f"all-passages-{result['id']}"):
                        for source in result["retrieved_sources"]:
                            st.write(f"**{source['title']} — PDF page {source['page']}**")
                            if source.get("dependency"):
                                st.caption(f"Linked evidence: {source['dependency']}")
                            if source.get("applicability"):
                                st.caption(f"Subclass scope: {source['applicability']}")
                            st.text(source["text"])
            if result.get('diagnostic_context'):
                reply=st.text_area('Current verified settings or observations',max_chars=500,
                    help='For example: DIP switch 1 = OFF; Function 40 = 0. You can also describe the event sequence.',
                    key=f"diagnostic-reply-{result['id']}")
                if st.button('Continue diagnosis',key=f"continue-{result['id']}",
                             disabled=not reply.strip()or active_pdf_trial):
                    try:
                        if contains_devanagari_letters(reply):
                            raise ValueError('Correct the observation to Bengali or English before continuing.')
                        with st.spinner('Checking the same case with your observations'):
                            retriever=get_retriever(settings.runtime,settings.embedding_model,
                                'cross-encoder/ms-marco-MiniLM-L6-v2'if use_reranker else None)
                            retriever.sync()
                            st.session_state['result']=continue_diagnosis(db,retriever,
                                ManualProvider(settings.base_url,settings.api_key,settings.model),result['id'],reply,
                                st.session_state.get('trial_id'))
                            st.session_state.pop('answer_audio',None)
                        st.rerun()
                    except (ValueError,ProviderError)as exc:
                        st.error(str(exc))
        with st.expander("Testing feedback and results"):
            if result:
                verdict = st.radio("Your source check", ["unclear", "correct", "incorrect"],
                                   horizontal=True, key=f"verdict-{result['id']}")
                note = st.text_area("What was missing or wrong? Include the expected answer/page if possible.",
                                     key=f"feedback-{result['id']}")
                if st.button("Save test feedback", key=f"save-feedback-{result['id']}"):
                    save_feedback(db, result["id"], verdict, note)
                    st.success("Feedback saved for the next correction cycle.")
            st.download_button("Download manual-testing results", export_feedback(db),
                               file_name="manual-testing-results.csv", mime="text/csv")

with study:
    st.subheader("Comparison session")
    st.caption("For synthetic rehearsals now. Do not use with participants until protocol, source review, "
               "privacy, and researcher access are approved.")
    trial_id = st.session_state.get("trial_id")
    if not trial_id:
        with db.connect() as connection:
            unfinished = connection.execute("SELECT id, condition, scenario FROM trials "
                                            "WHERE status='active'").fetchall()
        if unfinished:
            pending = unfinished[0]
            st.warning(f"Unfinished {pending['condition']} trial: {pending['scenario']}")
            if st.button("Resume unfinished trial"):
                st.session_state["trial_id"] = pending["id"]
                st.rerun()
            if st.button("Mark unfinished trial interrupted"):
                interrupt_trial(db, pending["id"])
                st.rerun()
        condition = st.radio("Condition", ["assistant", "pdf"], horizontal=True)
        scenario = st.text_area("Scenario shown to participant")
        if st.button("Start trial"):
            try:
                st.session_state["trial_id"] = start_trial(db, condition, scenario)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    else:
        with db.connect() as connection:
            trial = connection.execute("SELECT * FROM trials WHERE id=?", (trial_id,)).fetchone()
        if trial and trial["status"] == "active":
            st.write(f"**{trial['condition'].upper()}** • {trial['scenario']}")
            if st.button("Interrupt trial"):
                interrupt_trial(db, trial_id)
                st.session_state.pop("trial_id", None)
                st.rerun()
            if trial["condition"] == "pdf":
                st.info("Open the uploaded PDF in your normal PDF reader for this condition. "
                        "The assistant question screen remains available for development only.")
                chosen = st.selectbox("Manual to open", list(names) if manuals else [])
                if manuals and chosen:
                    manual_id = names[chosen]
                    st.download_button("Download selected PDF", (settings.runtime / "manuals" /
                                       f"{manual_id}.pdf").read_bytes(),
                                       file_name=next(m["filename"] for m in manuals if m["id"] == manual_id))
            response = st.text_area("Participant's final answer", key="final_response")
            if st.button("Submit final answer", disabled=not response.strip()):
                try:
                    finish_trial(db, trial_id, response)
                    st.session_state.pop("trial_id", None)
                    st.session_state.pop("final_response", None)
                    st.success("Final answer saved")
                except ValueError as exc:
                    st.error(str(exc))
        else:
            st.session_state.pop("trial_id", None)
    st.download_button("Export study CSV", export_study(db), file_name="study_trials.csv",
                       mime="text/csv")
    st.download_button("Export query evidence CSV", export_queries(db),
                       file_name="study_queries.csv", mime="text/csv")
