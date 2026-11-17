"""SQLite-backed vector store.

Chunks and their embeddings live in a single SQLite file
(``data/careerlens.db`` by default). Embeddings are stored as float32
blobs; search is exact cosine similarity computed with numpy — simple,
dependency-free, and fast enough for a corpus of this size.
"""

import sqlite3

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""


class VectorStore:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -- metadata ------------------------------------------------------
    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key, default=None):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    # -- chunks --------------------------------------------------------
    def clear(self):
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM meta")
        self.conn.commit()

    def add_chunks(self, chunks, embeddings, embedder_name):
        """Store chunks with their embedding rows (already L2-normalised)."""
        self.clear()
        for chunk, vec in zip(chunks, embeddings):
            blob = np.asarray(vec, dtype=np.float32).tobytes()
            self.conn.execute(
                "INSERT INTO chunks "
                "(doc_id, title, source, chunk_index, text, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chunk["doc_id"], chunk["title"], chunk["source"],
                 chunk["chunk_index"], chunk["text"], blob),
            )
        self.set_meta("embedder", embedder_name)
        self.set_meta("chunk_count", str(len(chunks)))
        self.conn.commit()

    def count(self):
        row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    def all_chunks(self):
        rows = self.conn.execute(
            "SELECT doc_id, title, source, chunk_index, text FROM chunks "
            "ORDER BY id"
        ).fetchall()
        return [
            {"doc_id": r[0], "title": r[1], "source": r[2],
             "chunk_index": r[3], "text": r[4]}
            for r in rows
        ]

    def search(self, query_vec, k=5):
        """Top-k chunks by cosine similarity (embeddings are normalised)."""
        rows = self.conn.execute(
            "SELECT doc_id, title, source, chunk_index, text, embedding "
            "FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            return []
        mat = np.stack([
            np.frombuffer(r[5], dtype=np.float32).astype(np.float64)
            for r in rows
        ])
        q = np.asarray(query_vec, dtype=np.float64)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = mat @ q
        order = np.argsort(-scores)[:k]
        results = []
        for idx in order:
            r = rows[idx]
            results.append({
                "doc_id": r[0], "title": r[1], "source": r[2],
                "chunk_index": r[3], "text": r[4],
                "score": float(scores[idx]),
            })
        return results
