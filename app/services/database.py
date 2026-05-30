import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.core.config import Settings


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.sqlite_path
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_base (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  owner_role TEXT NOT NULL DEFAULT 'TEACHER',
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  course_id TEXT NOT NULL DEFAULT '',
  embedding_model TEXT NOT NULL,
  embedding_dimensions INTEGER NOT NULL DEFAULT 1024,
  chunk_size INTEGER NOT NULL DEFAULT 512,
  chunk_overlap INTEGER NOT NULL DEFAULT 64,
  doc_visibility TEXT NOT NULL DEFAULT 'public',
  class_ids_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document (
  id TEXT PRIMARY KEY,
  knowledge_base_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  file_name TEXT NOT NULL,
  stored_path TEXT NOT NULL,
  content_type TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'processing',
  chunk_count INTEGER NOT NULL DEFAULT 0,
  token_count INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  error_message TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  processed_at TEXT,
  FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_base(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunk (
  id TEXT PRIMARY KEY,
  knowledge_base_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  token_count INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_base(id) ON DELETE CASCADE,
  FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunk_kb ON chunk(knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunk(document_id);

CREATE TABLE IF NOT EXISTS conversation (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  knowledge_base_ids_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_message (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  sources_json TEXT NOT NULL DEFAULT '[]',
  usage_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY (conversation_id) REFERENCES conversation(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS qa_log (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL DEFAULT '',
  knowledge_base_ids_json TEXT NOT NULL DEFAULT '[]',
  query TEXT NOT NULL,
  answer_text TEXT NOT NULL,
  sources_json TEXT NOT NULL DEFAULT '[]',
  top1_score REAL NOT NULL DEFAULT 0,
  coverage_score REAL NOT NULL DEFAULT 0,
  used_web INTEGER NOT NULL DEFAULT 0,
  feedback INTEGER,
  intent_type TEXT NOT NULL DEFAULT 'rag',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotation (
  id TEXT PRIMARY KEY,
  knowledge_base_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  annotation_type TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  owner_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_base(id) ON DELETE CASCADE,
  FOREIGN KEY (chunk_id) REFERENCES chunk(id) ON DELETE CASCADE
);
"""
