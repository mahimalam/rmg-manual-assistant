"""SQLite is the source of truth for uploaded documents and study events."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS manuals (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE, pages INTEGER NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, manual_id TEXT NOT NULL REFERENCES manuals(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL,
                    extraction TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_manual ON chunks(manual_id);
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, question TEXT NOT NULL,
                    search_question TEXT, answer TEXT NOT NULL, status TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    trial_id TEXT REFERENCES trials(id)
                );
                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, condition TEXT NOT NULL,
                    scenario TEXT NOT NULL, status TEXT NOT NULL,
                    final_answer TEXT, ended_at TEXT, elapsed_seconds REAL
                );
                CREATE TABLE IF NOT EXISTS knowledge_units (
                    id TEXT PRIMARY KEY,
                    manual_id TEXT NOT NULL REFERENCES manuals(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
                    payload_json TEXT NOT NULL, release_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS units_manual ON knowledge_units(manual_id);
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    manual_id TEXT PRIMARY KEY REFERENCES manuals(id) ON DELETE CASCADE,
                    status TEXT NOT NULL, pages_done INTEGER NOT NULL DEFAULT 0,
                    release_id TEXT, error TEXT, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS extraction_attempts (
                    id TEXT PRIMARY KEY, cache_key TEXT NOT NULL UNIQUE,
                    manual_id TEXT REFERENCES manuals(id) ON DELETE SET NULL,
                    status TEXT NOT NULL, reserved_usd REAL NOT NULL,
                    actual_usd REAL, usage_json TEXT, model TEXT NOT NULL,
                    created_at TEXT NOT NULL, error TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    query_id TEXT PRIMARY KEY REFERENCES queries(id) ON DELETE CASCADE,
                    verdict TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(queries)")}
            if "search_question" not in columns:
                db.execute("ALTER TABLE queries ADD COLUMN search_question TEXT")
            for column, declaration in (("evidence_json", "TEXT NOT NULL DEFAULT '[]'"),
                                        ("provider_usage_json", "TEXT NOT NULL DEFAULT '[]'"),
                                        ("timings_json", "TEXT NOT NULL DEFAULT '{}'"),
                                        ("retrieval_release", "TEXT"),
                                        ("error", "TEXT"),
                                        ("provider_responses_json", "TEXT NOT NULL DEFAULT '[]'"),
                                        ("translation_issue", "TEXT"),
                                        ("diagnostic_context_json", "TEXT"),
                                        ("input_context_json", "TEXT")):
                if column not in columns:
                    db.execute(f"ALTER TABLE queries ADD COLUMN {column} {declaration}")
            manual_columns = {row[1] for row in db.execute("PRAGMA table_info(manuals)")}
            for column, declaration in (("knowledge_release", "TEXT"),
                                        ("knowledge_status", "TEXT NOT NULL DEFAULT 'baseline'")):
                if column not in manual_columns:
                    db.execute(f"ALTER TABLE manuals ADD COLUMN {column} {declaration}")

    def manuals(self):
        with self.connect() as db:
            return db.execute("SELECT * FROM manuals ORDER BY created_at DESC").fetchall()

    def chunks(self, manual_ids=None):
        with self.connect() as db:
            if manual_ids:
                placeholders = ",".join("?" for _ in manual_ids)
                return db.execute(
                    f"SELECT c.*, m.title,m.knowledge_release FROM chunks c JOIN manuals m ON m.id=c.manual_id "
                    f"WHERE m.status='ready' AND c.manual_id IN ({placeholders})",
                    tuple(manual_ids),
                ).fetchall()
            return db.execute(
                "SELECT c.*, m.title,m.knowledge_release FROM chunks c JOIN manuals m ON m.id=c.manual_id "
                "WHERE m.status='ready'"
            ).fetchall()

    def units(self, manual_id=None):
        with self.connect() as db:
            if manual_id:
                return db.execute("SELECT * FROM knowledge_units WHERE manual_id=? ORDER BY page,id",
                                  (manual_id,)).fetchall()
            return db.execute("SELECT * FROM knowledge_units ORDER BY manual_id,page,id").fetchall()

    def job(self, manual_id):
        with self.connect() as db:
            return db.execute("SELECT * FROM ingestion_jobs WHERE manual_id=?", (manual_id,)).fetchone()

    def ingestion_spend(self):
        with self.connect() as db:
            row = db.execute("SELECT COUNT(*), COALESCE(SUM(COALESCE(actual_usd,reserved_usd)),0) "
                             "FROM extraction_attempts").fetchone()
            return {"attempts": row[0], "accounted_usd": row[1]}
