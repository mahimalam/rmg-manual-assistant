"""Persistent local vectors, independent BM25, rank fusion and evidence closure."""

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.structured import atomic_json, digest, manual_lock, _searchable
from app.semantics import extract_subclasses, matches_model
from app.conditions import pitch_condition
from app.language import model_ids
from app.questions import question_parts, diagnostic_question, event_trigger, audible_noise_question, electrical_noise_only, unrelated_noise_setting
from app.model_loading import serialized_model_load


@serialized_model_load
@lru_cache(maxsize=2)
def encoder(model_name: str):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(4)

    try:
        snapshot = model_name if Path(model_name).is_dir() else snapshot_download(model_name, local_files_only=True)
        return SentenceTransformer(snapshot, device="cpu", local_files_only=True)
    except OSError:
        snapshot = snapshot_download(model_name, allow_patterns=["*.json", "*.bin", "*.model"])
        return SentenceTransformer(snapshot, device="cpu", local_files_only=True)


@serialized_model_load
@lru_cache(maxsize=1)
def reranker(model_name):
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder
    import torch

    torch.set_num_threads(4)

    files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.txt"]
    try:
        snapshot = snapshot_download(model_name, local_files_only=True, allow_patterns=files)
    except OSError:
        snapshot = None
    if snapshot is None or not (Path(snapshot) / "model.safetensors").exists():
        snapshot = snapshot_download(model_name, allow_patterns=files, max_workers=2)
    return CrossEncoder(snapshot, device="cpu", max_length=512, local_files_only=True)


STOPWORDS = {"a", "an", "and", "are", "be", "before", "can", "do", "does", "for", "how",
             "if", "in", "is", "it", "of", "on", "should", "the", "to", "what", "when",
             "where", "which", "with", "you"}


def tokens(text):
    terms = [term for term in re.findall(r"[\w]+", text.lower()) if term not in STOPWORDS]
    # Match singular/plural English manual terms without changing identifiers.
    return [term[:-1] if len(term) > 3 and term.isascii() and term.isalpha() and
            term.endswith('s') and not term.endswith('ss') else term for term in terms]


def lexical_rank(question, rows):
    """BM25 searches the entire eligible corpus, independent of vector candidates."""
    if not rows:
        return []
    query_terms = set(tokens(question))
    documents = [Counter(tokens(row["text"])) for row in rows]
    lengths = [sum(document.values()) for document in documents]
    average = sum(lengths) / len(lengths) or 1
    frequencies = Counter(term for document in documents for term in document)
    scores = []
    for row, document, length in zip(rows, documents, lengths):
        score = 0.0
        for term in query_terms:
            frequency = document[term]
            if frequency:
                idf = math.log(1 + (len(rows) - frequencies[term] + 0.5) /
                               (frequencies[term] + 0.5))
                score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            scores.append((row["id"], score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def model_signature(model_name):
    reference = Path.home() / ".cache/huggingface/hub" / (
        "models--" + model_name.replace("/", "--")) / "refs/main"
    revision = reference.read_text().strip() if reference.exists() else "unresolved"
    return f"{model_name}@{revision}"


class Retriever:
    def __init__(self, db, runtime: Path, model_name: str, reranker_model=None):
        self.db, self.runtime, self.model_name = db, Path(runtime), model_name
        self.signature = model_signature(model_name)
        self.reranker_model = reranker_model
        import chromadb

        self.client = chromadb.PersistentClient(path=str(self.runtime / "index"))
        self.pointer = self.runtime / "index" / "active.json"
        if self.pointer.exists():
            manifest = json.loads(self.pointer.read_text())
            self.collection = self.client.get_collection(manifest["collection"], embedding_function=None)
            self.release_id = manifest["release_id"]
            self.model_mismatch = manifest["model_signature"] != self.signature
        else:
            self.collection = self.client.get_or_create_collection(
                "manual_chunks_v1", configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None)
            self.release_id, self.model_mismatch = "baseline", False

    def _encode(self, texts):
        model = encoder(self.model_name)
        for text in texts:
            if len(model.tokenizer(text, truncation=False)["input_ids"]) > model.max_seq_length:
                raise ValueError("A knowledge unit exceeds the embedding tokenizer limit; split it before indexing")
        return [[float(value) for value in vector]
                for vector in model.encode(texts, normalize_embeddings=True, batch_size=8)]

    def _cached_vectors(self, rows):
        vectors, missing = {}, []
        for row in rows:
            key = digest([self.signature, "normalized-v1", row["text"]])
            path = self.runtime / "embeddings" / row["manual_id"] / f"{key}.json"
            if path.exists():
                vectors[row["id"]] = json.loads(path.read_text())
            else:
                missing.append((row, path))
        for start in range(0, len(missing), 24):
            batch = missing[start:start + 24]
            encoded = self._encode([row["text"] for row, _ in batch])
            for (row, path), vector in zip(batch, encoded):
                atomic_json(path, vector)
                vectors[row["id"]] = vector
        for vector in vectors.values():
            if (not isinstance(vector, list) or not vector or
                    any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in vector) or
                    abs(sum(value * value for value in vector) - 1.0) > 0.02):
                raise ValueError("Cached embedding is invalid; repair that cache before indexing")
        return [vectors[row["id"]] for row in rows]

    def sync(self, progress=None):
        with manual_lock(self.runtime, "vector-index"):
            return self._sync(progress)

    def _corpus_digest(self, rows):
        return digest([self.signature, sorted((row["id"], row["text"], row.get("knowledge_release"))
                                              for row in rows)])[:24]

    def _stage(self, rows, progress=None):
        if rows and self.signature.endswith('@unresolved'):
            encoder(self.model_name)
            self.signature=model_signature(self.model_name)
        release = self._corpus_digest(rows)
        name = f"units_{release}"
        collection = self.client.get_or_create_collection(
            name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
            metadata={"model_signature": self.signature, "release_id": release})
        indexed = set(collection.get(include=[])["ids"])
        missing = sorted((row for row in rows if row["id"] not in indexed),
                         key=lambda row: len(row["text"]))
        for start in range(0, len(missing), 24):
            batch = missing[start:start + 24]
            collection.upsert(ids=[row["id"] for row in batch],
                              embeddings=self._cached_vectors(batch),
                              metadatas=[{"manual_id": row["manual_id"], "page": row["page"]}
                                         for row in batch])
            if progress:
                progress(collection.count(), len(rows))
        if set(collection.get(include=[])["ids"]) != {row["id"] for row in rows}:
            raise ValueError("Staged vector count does not match the knowledge release")
        return {"collection": name, "release_id": release, "model_signature": self.signature,
                "records": len(rows), "metric": "cosine", "retrieval": "bm25-rrf-v1"}

    def publish_prepared(self, manifest):
        current = [dict(row) for row in self.db.chunks()]
        if self._corpus_digest(current) != manifest["release_id"]:
            raise ValueError("Library changed during indexing; retry to publish the current release")
        atomic_json(self.pointer, manifest)
        self.collection = self.client.get_collection(manifest["collection"], embedding_function=None)
        self.release_id = manifest["release_id"]

    def _sync(self, progress=None):
        if self.model_mismatch:
            raise ValueError("The embedding model changed. Rebuild the index before asking questions.")
        rows = [dict(row) for row in self.db.chunks()]
        if self.release_id == self._corpus_digest(rows) and self.collection.count() == len(rows):
            return len(rows)
        manifest = self._stage(rows, progress)
        self.publish_prepared(manifest)
        return len(rows)

    def prepare(self, manual_id, units, release_id, progress=None):
        """Build replacement vectors before the caller changes published source rows."""
        if self.model_mismatch:
            raise ValueError("Rebuild the changed embedding model before publishing knowledge")
        with manual_lock(self.runtime, "vector-index"):
            rows = [dict(row) for row in self.db.chunks() if row["manual_id"] != manual_id]
            rows.extend({"id": unit["id"], "manual_id": manual_id, "page": unit["page"],
                         "text": unit["search_text"], "knowledge_release": release_id}
                        for unit in units if _searchable(unit))
            return self._stage(rows, progress)

    def search(self, question: str, manual_ids=None, limit: int = 4):
        queries=question_parts(question)
        if len(queries)==1:
            evidence=self._search_one(question,manual_ids,limit)
            if diagnostic_question(question):
                for source in evidence:
                    source.update(retrieval_queries=queries,retrieval_intents=[0],
                                  primary_intents=[] if source.get('dependency') else [0],require_claims=True)
            return evidence
        result={}
        for intent,query in enumerate(queries):
            evidence=self._search_one(query,manual_ids,limit=limit)
            for source in evidence:
                existing=result.setdefault(source['id'],{**source,'retrieval_intents':[],
                                                         'primary_intents':[],
                                                         'retrieval_queries':queries})
                if intent not in existing['retrieval_intents']:
                    existing['retrieval_intents'].append(intent)
                if not source.get('dependency') and intent not in existing['primary_intents']:
                    existing['primary_intents'].append(intent)
                    existing.pop('dependency',None)
        if sum(len(s['text']) for s in result.values())>30000:
            raise ValueError('Combined symptom evidence is too large; narrow the question')
        return list(result.values())

    def _search_one(self, question: str, manual_ids=None, limit: int = 4):
        if self.model_mismatch:
            raise ValueError("The embedding model changed. Rebuild the index before asking questions.")
        rows = [dict(row) for row in self.db.chunks(manual_ids)]
        if not question.strip() or not rows:
            return []
        lookup = {row["id"]: row for row in rows}
        where = {"manual_id": {"$in": list(manual_ids)}} if manual_ids else None
        dense, similarities = [], {}
        if self.collection.count():
            result = self.collection.query(query_embeddings=self._encode([question]),
                                           n_results=min(24, len(rows)), where=where,
                                           include=["distances"])
            for key, distance in zip(result["ids"][0], result["distances"][0]):
                score = 1 - float(distance)
                if not math.isfinite(score) or score < -1.01 or score > 1.01:
                    raise ValueError("Invalid cosine score in the index")
                if key in lookup:
                    dense.append(key)
                    similarities[key] = score
        lexical = lexical_rank(question, rows)[:24]
        fused = Counter()
        for ranking in (dense, [key for key, _ in lexical]):
            for rank, key in enumerate(ranking, 1):
                fused[key] += 1 / (60 + rank)
        lexical_scores = dict(lexical)
        matches = []
        candidates = sorted(fused, key=fused.get, reverse=True)[:24]
        if audible_noise_question(question):
            candidates = [key for key in candidates if not electrical_noise_only(lookup[key]['text'])
                          and not unrelated_noise_setting(question, lookup[key]['text'])]
        reranked = {}
        truncated = {}
        apply_reranker = bool(self.reranker_model and candidates and not re.search(r'[\u0980-\u09ff]', question))
        if apply_reranker:
            model = reranker(self.reranker_model)
            documents = {key: f"Manual: {lookup[key]['title']}\n{lookup[key]['text']}"
                         for key in candidates}
            for key in candidates:
                truncated[key] = len(model.tokenizer(question, documents[key],
                                                     truncation=False)["input_ids"]) > 512
            values = model.predict([(question, documents[key]) for key in candidates],
                                   batch_size=8, show_progress_bar=False)
            reranked = {key: float(score) for key, score in zip(candidates, values)}
            if any(not math.isfinite(value) for value in reranked.values()):
                raise ValueError("The evidence reranker returned invalid scores")
            candidates.sort(key=reranked.get, reverse=True)
        trigger=event_trigger(question)
        trigger_matches=set()
        if trigger and candidates:
            with self.db.connect()as connection:
                records=connection.execute('SELECT id,payload_json FROM knowledge_units WHERE id IN ('+
                    ','.join('?'for _ in candidates)+')',candidates).fetchall()
            trigger_matches={record['id']for record in records if any(rule['effect']and rule['trigger']==trigger for rule in
                             json.loads(record['payload_json']).get('setting_rules',[]))}
            candidates.sort(key=lambda key:key not in trigger_matches)
        for key in candidates[:limit]:
            row = lookup[key]
            matches.append({"id": key, "manual_id": row["manual_id"], "title": row["title"],
                            "page": row["page"], "text": row["text"],
                            "score": similarities.get(key, 0.0), "rank_score": fused[key],
                            "lexical_score": lexical_scores.get(key, 0.0),
                            "corpus_release": row.get("knowledge_release"),
                            "reranker_model": self.reranker_model,
                            "reranker_revision": model_signature(self.reranker_model) if self.reranker_model else None,
                            "reranker_score": reranked.get(key),
                            "reranker_applied": apply_reranker,
                            "reranker_input_truncated": truncated.get(key, False),
                            "release_id": self.release_id})
            matches[-1]['source_trigger_matched']=key in trigger_matches
        evidence = self._expand(matches, lookup)
        subclasses = extract_subclasses(question)
        if len(subclasses)==1:
            scoped=[s for s in evidence if not s.get('dependency') and
                    pitch_condition(question,s['text']) is True and
                    any(matches_model(pattern,subclasses[0]) for pattern in s.get('applicable_models',[]))]
            if scoped:
                # A matching model column and explicit quantity condition take
                # precedence over different-mode passages from the same manual.
                evidence=self._expand(scoped,lookup)
        evidence=[source for source in evidence if source.get('dependency') or
                  pitch_condition(question,source['text']) is not False]
        for source in evidence:
            comparison=pitch_condition(question,source['text'])
            if comparison is not None:
                source['query_pitch_condition_satisfied']=comparison
            source["requested_subclasses"] = subclasses
            source["applicability"] = {}
            for subclass in subclasses:
                if any(matches_model(pattern, subclass) for pattern in source.get("excluded_models", [])):
                    status = "explicitly_excluded"
                elif source.get("applicable_models"):
                    status = ("within_declared_scope" if any(matches_model(pattern, subclass)
                              for pattern in source["applicable_models"]) else "outside_declared_scope")
                else:
                    status = "not_established"
                source["applicability"][subclass] = status
        return evidence

    def _expand(self, matches, lookup):
        units = {row["id"]: json.loads(row["payload_json"]) for row in self.db.units()}
        authorized = {row["manual_id"]: row["title"] for row in lookup.values()}
        lookup = dict(lookup)
        for key, unit in units.items():
            if key not in lookup and unit["manual_id"] in authorized:
                # Row evidence already contains its cells. Parent context supplies
                # headers, conditions, warnings and footnotes without unrelated rows.
                text = unit["search_text"]
                if unit["kind"] == "table":
                    parts = [unit["title"], "Column headers: " + "; ".join(unit["table_headers"])]
                    if unit["condition"]:
                        parts.append("Condition: " + unit["condition"])
                    from app.semantics import note_numbers
                    general_notes=[n for n in unit['footnotes'] if not note_numbers(n)]
                    for label, values in (("Prerequisite", unit["prerequisites"]),
                                          ("Warning", unit["warnings"]), ("Footnote", general_notes)):
                        parts.extend(f"{label}: {value}" for value in values)
                    text = "\n".join(parts)
                lookup[key] = {"id": key, "manual_id": unit["manual_id"], "page": unit["page"],
                               "title": authorized[unit["manual_id"]], "text": text}
        # Expand small hits to bounded related prose on the same physical page.
        # Sibling subsections often split a duration from its repeat/exception.
        context_sources, context_ids, page_sizes = [], {s['id'] for s in matches}, {}
        context_size=0
        for source in matches:
            owner=units.get(source['id'],{})
            section=owner.get('section_key','')
            if owner.get('kind')=='table' or owner.get('setting_id') or owner.get('note_ids') or not section:
                continue
            scope=section.rsplit('-',1)[0] if '-' in section else section
            for key,unit in units.items():
                other=unit.get('section_key','')
                other_scope=other.rsplit('-',1)[0] if '-' in other else other
                if (key in context_ids or key not in lookup or unit['manual_id']!=source['manual_id'] or
                        unit['page']!=source['page'] or unit['kind']=='table' or unit['blocking_issues'] or
                        other_scope!=scope):
                    continue
                page_key=(source['manual_id'],source['page'])
                size=page_sizes.get(page_key,0)+len(lookup[key]['text'])
                if size>4000 or context_size+len(lookup[key]['text'])>4000:
                    continue
                page_sizes[page_key]=size;context_ids.add(key)
                context_size+=len(lookup[key]['text'])
                context_sources.append({**lookup[key],'score':0.0,'dependency':'context','release_id':self.release_id})
        result, seen, visited, queue = [], set(), set(), list(matches)+context_sources
        while queue:
            source = queue.pop(0)
            context = (source["id"], source.get("required_value"))
            if context in visited:
                continue
            visited.add(context)
            unit = units.get(source["id"], {})
            if unit.get("blocking_issues"):
                if source.get('dependency') == 'procedure':
                    owner = next(item for item in result if item['id'] == source['dependency_owner'])
                    owner.setdefault('unresolved_dependencies', []).append({
                        'page': source['page'], 'relation': 'procedure',
                        'issues': unit['blocking_issues']})
                    continue
                raise ValueError("Required evidence has unresolved extraction issues; inspect the source page")
            if source["id"] not in seen:
                seen.add(source["id"])
                metadata = {key: unit.get(key) for key in ("setting_id", "setting_value", "setting_requirements", "setting_exclusions", "setting_rules",
                            "applicable_models", "excluded_models", "is_applicability_rule", "section_key",
                            "note_ids", "note_references", "source_refs", "warnings") if key in unit}
                if unit.get('kind')=='table':
                    metadata['table_model_aliases']={header:model_ids(header) for header in unit['table_headers']
                                                     if model_ids(header)}
                flags=unit.get('flags',[])
                if unit.get('kind')=='table' and not unit.get('parent_id') and len(unit.get('table_rows',[]))>1:
                    # The excerpt contains headers/notes, not the full-table quote.
                    # Row-level quotation checks remain attached to actual cells.
                    flags=[flag for flag in flags if flag!='Visual quotation does not match the text layer; source review needed']
                    if all('header unresolved'not in h for h in unit['table_headers']):
                        # Only headers are rendered; an uncertain sibling row
                        # is not evidence here and cannot taint a complete row.
                        flags=[flag for flag in flags if flag!='Merged/blank cells or headers require visual source review']
                result.append({**source, **metadata, "kind": unit.get("kind", "information"),
                               "review_status": unit.get("review_status", "unreviewed"),
                               "flags": flags, "required_evidence_ids": []})
            for dependency in unit.get("dependencies", []):
                if source.get('dependency')=='context' and dependency['relation']=='procedure':
                    continue  # An optional sibling must not recursively import unrelated pages.
                if (dependency.get("when_value") is not None and source.get("required_value") is not None
                        and dependency["when_value"] != source["required_value"]):
                    continue
                key = dependency["id"]
                if key not in lookup or lookup[key]["manual_id"] != source["manual_id"]:
                    raise ValueError("Required evidence is missing or outside the selected manual")
                if dependency["relation"] in ("note", "prerequisite", "applicability", "warning"):
                    owner = next(item for item in result if item["id"] == source["id"])
                    if key not in owner["required_evidence_ids"]:
                        owner["required_evidence_ids"].append(key)
                if (key, dependency.get("required_value")) not in visited:
                    row = lookup[key]
                    queue.append({"id": key, "manual_id": row["manual_id"], "title": row["title"],
                                  "page": row["page"], "text": row["text"], "score": 0.0,
                                  "dependency": dependency["relation"], "release_id": self.release_id,
                                  "dependency_owner": source['id'],
                                  "required_value": dependency.get("required_value")})
            if sum(len(item["text"]) for item in result) > 30000:
                raise ValueError("Required evidence is too large; narrow the question or select a source page")
        return result

    def rebuild(self):
        self.signature = model_signature(self.model_name)
        self.model_mismatch = False
        self.release_id = "rebuild"
        return self.sync()

    def remove_manual(self, manual_id: str):
        for collection in self.client.list_collections():
            records = self.client.get_collection(collection.name, embedding_function=None)
            ids = records.get(where={"manual_id": manual_id}, include=[])["ids"]
            if ids:
                records.delete(ids=ids)
        return self.sync()
