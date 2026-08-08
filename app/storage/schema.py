from __future__ import annotations


SCHEMA_VERSION = 1

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        corpus_namespace TEXT NOT NULL
            CHECK (corpus_namespace IN ('demo', 'uploaded', 'evaluation')),
        original_filename TEXT NOT NULL,
        stored_filename TEXT,
        file_type TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        source TEXT,
        content_hash TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        status TEXT NOT NULL,
        chunk_size INTEGER NOT NULL,
        chunk_overlap INTEGER NOT NULL,
        chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error_code TEXT,
        error_message TEXT,
        index_version INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS documents_namespace_content_uq
    ON documents(corpus_namespace, content_hash)
    """,
    """
    CREATE INDEX IF NOT EXISTS documents_namespace_status_idx
    ON documents(corpus_namespace, status, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL
            REFERENCES documents(document_id) ON DELETE CASCADE,
        corpus_namespace TEXT NOT NULL
            CHECK (corpus_namespace IN ('demo', 'uploaded', 'evaluation')),
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        title TEXT NOT NULL,
        section TEXT,
        page_number INTEGER,
        text TEXT NOT NULL CHECK (length(trim(text)) > 0),
        content_hash TEXT NOT NULL,
        token_count INTEGER,
        created_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(document_id, chunk_index)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS chunks_document_idx
    ON chunks(document_id, chunk_index)
    """,
    """
    CREATE INDEX IF NOT EXISTS chunks_namespace_idx
    ON chunks(corpus_namespace)
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_jobs (
        job_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL
            REFERENCES documents(document_id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        current_stage TEXT NOT NULL,
        progress INTEGER NOT NULL CHECK (progress BETWEEN 0 AND 100),
        error_code TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS ingestion_jobs_document_idx
    ON ingestion_jobs(document_id, created_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id TEXT NOT NULL
            REFERENCES chunks(chunk_id) ON DELETE CASCADE,
        model_name TEXT NOT NULL,
        dimension INTEGER NOT NULL CHECK (dimension > 0),
        normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
        content_hash TEXT NOT NULL,
        vector_blob BLOB NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(chunk_id, model_name, normalized)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS index_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        current_version INTEGER NOT NULL DEFAULT 0,
        schema_version INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)
