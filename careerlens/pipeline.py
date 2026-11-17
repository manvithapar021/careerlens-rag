"""Pipeline: corpus loading, index building, and end-to-end analysis.

Flow: ingest (markdown docs in ``data/corpus``) -> chunk -> embed ->
SQLite vector store -> retrieve top-k -> classify JD skills against
corpus evidence -> compose a grounded, cited fit analysis.
"""

from pathlib import Path

from . import chunking, composer, embeddings, skills
from .store import VectorStore

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "careerlens.db"


def load_corpus(corpus_dir=DEFAULT_CORPUS_DIR):
    """Load every ``*.md`` file in ``corpus_dir`` as a document."""
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = path.stem.replace("-", " ").replace("_", " ").title()
        body_lines = []
        for line in raw.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                continue  # heading is metadata, not chunk content
            body_lines.append(line)
        text = "\n".join(body_lines).strip()
        source = "resume" if path.stem == "resume" else f"project: {path.stem}"
        docs.append({
            "doc_id": path.stem, "title": title,
            "source": source, "text": text,
        })
    return docs


def build_index(corpus_dir=DEFAULT_CORPUS_DIR, db_path=DEFAULT_DB_PATH,
                embedder=None):
    """(Re)build the vector index from the corpus. Returns chunk count."""
    embedder = embedder or embeddings.get_embedder()
    docs = load_corpus(corpus_dir)
    chunks = []
    for doc in docs:
        chunks.extend(chunking.chunk_document(
            doc["doc_id"], doc["title"], doc["source"], doc["text"]))
    indexed_texts = [c["indexed_text"] for c in chunks]
    embedder.fit(indexed_texts)
    vectors = embedder.encode(indexed_texts)
    store = VectorStore(db_path)
    try:
        store.add_chunks(chunks, vectors, embedder.name)
        return len(chunks)
    finally:
        store.close()


def open_store(db_path=DEFAULT_DB_PATH, corpus_dir=DEFAULT_CORPUS_DIR,
               embedder=None):
    """Open the store, rebuilding it when it is empty or stale."""
    store = VectorStore(db_path)
    if store.count() == 0:
        store.close()
        build_index(corpus_dir, db_path, embedder)
        store = VectorStore(db_path)
    return store


def analyze(jd_text, db_path=DEFAULT_DB_PATH,
            corpus_dir=DEFAULT_CORPUS_DIR, top_k=5, use_llm=None,
            embedder=None):
    """Full analysis of one job description. Returns a result dict.

    ``use_llm``: True forces the hosted LLM (falls back to extractive on
    error), False forces extractive, None (default) uses the LLM only
    when one is configured via environment variables.
    """
    jd_text = (jd_text or "").strip()
    if not jd_text:
        raise ValueError("Job description is empty")
    embedder = embedder or embeddings.get_embedder()
    store = open_store(db_path, corpus_dir, embedder)
    try:
        # The index must have been built with this same embedder, or the
        # vector dimensions will not match. Rebuild when they differ.
        if store.get_meta("embedder") != embedder.name:
            store.close()
            build_index(corpus_dir, db_path, embedder)
            store = VectorStore(db_path)
        # A fresh TF-IDF embedder needs the corpus vocabulary before it
        # can encode a query (fitting is deterministic on the same corpus).
        if isinstance(embedder, embeddings.TfidfEmbedder) \
                and embedder.idf is None:
            docs = load_corpus(corpus_dir)
            texts = []
            for doc in docs:
                texts.extend(
                    c["indexed_text"] for c in chunking.chunk_document(
                        doc["doc_id"], doc["title"], doc["source"],
                        doc["text"]))
            embedder.fit(texts)
        query_vec = embedder.encode([jd_text])[0]
        retrieved = store.search(query_vec, k=top_k)
        all_chunks = store.all_chunks()
    finally:
        store.close()

    jd_skills = skills.extract_skills(jd_text)
    matched, gaps = skills.classify_skills(jd_skills, all_chunks)

    # Citations: for each matched skill, the best retrieved chunk that
    # evidences it (falling back to any corpus chunk that does).
    citations = {}
    for skill in matched:
        support = skills.supporting_chunks(skill, retrieved)
        if not support:
            support = skills.supporting_chunks(skill, all_chunks)
        if support:
            best = support[0]
            citations[skill] = {
                "doc_id": best["doc_id"], "title": best["title"],
                "source": best["source"],
                "score": next(
                    (r["score"] for r in retrieved
                     if r["doc_id"] == best["doc_id"]
                     and r["chunk_index"] == best["chunk_index"]),
                    None),
                "snippet": best["text"][:240],
            }

    result = composer.compose_extractive(
        jd_text, matched, gaps, retrieved, jd_skills)
    result.update({
        "jd_skills": jd_skills,
        "citations": citations,
        "retrieved": retrieved,
        "embedder": embedder.name,
    })

    want_llm = composer.llm_configured() is not None if use_llm is None \
        else bool(use_llm)
    if want_llm:
        try:
            llm = composer.compose_with_llm(jd_text, retrieved)
            result["mode"] = llm["mode"]
            result["narrative"] = llm["narrative"]
        except composer.ComposeError as exc:
            result["llm_error"] = str(exc)
            # Extractive result stands; the failure is reported, not hidden.
    return result
