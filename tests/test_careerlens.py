"""Tests for CareerLens — all use the TF-IDF embedder and a temp DB,
so they run offline, deterministically, and without any API key."""

from pathlib import Path

import pytest

from careerlens import chunking, composer, embeddings, pipeline, skills
from careerlens.store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"


@pytest.fixture()
def tfidf():
    return embeddings.TfidfEmbedder()


@pytest.fixture()
def built_store(tmp_path, tfidf):
    docs = pipeline.load_corpus(CORPUS)
    chunks = []
    for doc in docs:
        chunks.extend(chunking.chunk_document(
            doc["doc_id"], doc["title"], doc["source"], doc["text"]))
    texts = [c["indexed_text"] for c in chunks]
    tfidf.fit(texts)
    vectors = tfidf.encode(texts)
    store = VectorStore(tmp_path / "test.db")
    store.add_chunks(chunks, vectors, tfidf.name)
    yield store, chunks
    store.close()


# -- chunking ----------------------------------------------------------
def test_chunking_splits_long_text():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunking.chunk_text(text, chunk_words=120, overlap_words=30)
    # starts at 0, 90, 180, 270, 360, 450 -> 6 windows
    assert len(chunks) == 6
    assert all(len(c.split()) <= 120 for c in chunks)


def test_chunking_overlap_shares_words():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunking.chunk_text(text, chunk_words=120, overlap_words=30)
    first_tail = set(chunks[0].split()[-30:])
    second_head = set(chunks[1].split()[:30])
    assert first_tail & second_head, "adjacent chunks must overlap"


def test_chunking_short_text_is_one_chunk():
    assert len(chunking.chunk_text("hello world")) == 1
    assert chunking.chunk_text("") == []


# -- embeddings --------------------------------------------------------
def test_tfidf_embeddings_are_normalised(tfidf):
    tfidf.fit(["flask web app", "pandas data analysis", "flask sqlalchemy"])
    vecs = tfidf.encode(["flask app", "unknown words here"])
    assert vecs.shape[0] == 2
    import numpy as np
    norms = np.linalg.norm(vecs, axis=1)
    assert norms[0] == pytest.approx(1.0)
    # A query with no known words is the zero vector.
    assert norms[1] == pytest.approx(0.0)


# -- retrieval ---------------------------------------------------------
def test_retrieval_flask_query_returns_flask_projects_first(built_store):
    store, _ = built_store
    # Rebuild a TF-IDF embedder fitted on the corpus for the query.
    tfidf = embeddings.TfidfEmbedder()
    docs = pipeline.load_corpus(CORPUS)
    chunks = []
    for doc in docs:
        chunks.extend(chunking.chunk_document(
            doc["doc_id"], doc["title"], doc["source"], doc["text"]))
    tfidf.fit([c["indexed_text"] for c in chunks])
    q = tfidf.encode(["Flask web application with SQLAlchemy"])[0]
    results = store.search(q, k=3)
    assert results, "retrieval returned nothing"
    top_ids = [r["doc_id"] for r in results]
    assert top_ids[0] in ("safecampus", "event-rsvp", "secure-file-vault"), \
        f"expected a Flask project first, got {top_ids}"
    assert "flask" in results[0]["text"].lower()


def test_retrieval_scores_descend(built_store):
    store, _ = built_store
    tfidf = embeddings.TfidfEmbedder()
    docs = pipeline.load_corpus(CORPUS)
    chunks = []
    for doc in docs:
        chunks.extend(chunking.chunk_document(
            doc["doc_id"], doc["title"], doc["source"], doc["text"]))
    tfidf.fit([c["indexed_text"] for c in chunks])
    q = tfidf.encode(["Django REST API library loans"])[0]
    results = store.search(q, k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_store_roundtrip(tmp_path):
    store = VectorStore(tmp_path / "rt.db")
    try:
        chunks = [{"doc_id": "d1", "title": "Doc", "source": "resume",
                   "chunk_index": 0, "text": "hello flask world"}]
        emb = embeddings.TfidfEmbedder().fit(["hello flask world"])
        store.add_chunks(chunks, emb.encode(["hello flask world"]), "tfidf")
        assert store.count() == 1
        assert store.all_chunks()[0]["text"] == "hello flask world"
        assert store.get_meta("embedder") == "tfidf"
    finally:
        store.close()


# -- skills & honesty --------------------------------------------------
def test_gap_detection_react_is_a_gap_not_a_skill(tmp_path):
    jd = ("We need a React developer with Node.js and TypeScript, "
          "plus Python and Flask experience.")
    result = pipeline.analyze(jd, db_path=str(tmp_path / "a.db"),
                               embedder=embeddings.TfidfEmbedder())
    assert "React" in result["gaps"]
    assert "Node.js" in result["gaps"]
    assert "TypeScript" in result["gaps"]
    assert "React" not in result["matched"]
    assert "Python" in result["matched"]
    assert "Flask" in result["matched"]


def test_gap_detection_never_claims_docker_or_kubernetes(tmp_path):
    jd = "DevOps intern: Docker, Kubernetes, AWS and CI/CD pipelines."
    result = pipeline.analyze(jd, db_path=str(tmp_path / "b.db"),
                               embedder=embeddings.TfidfEmbedder())
    for skill in ("Docker", "Kubernetes", "AWS", "CI/CD"):
        assert skill in result["gaps"], f"{skill} should be a gap"
        assert skill not in result["matched"]


def test_matched_skill_has_citation(tmp_path):
    jd = ("Backend Python intern: Flask, REST APIs, SQL, Git. "
          "Build CRUD services with SQLAlchemy.")
    result = pipeline.analyze(jd, db_path=str(tmp_path / "c.db"),
                               embedder=embeddings.TfidfEmbedder())
    assert result["matched"], "expected matched skills"
    for skill in result["matched"]:
        cit = result["citations"].get(skill)
        assert cit is not None, f"{skill} has no citation"
        assert cit["doc_id"] and cit["title"] and cit["snippet"]


def test_citations_present_in_retrieved_chunks(tmp_path):
    jd = "Django REST Framework API with token authentication."
    result = pipeline.analyze(jd, db_path=str(tmp_path / "d.db"),
                               embedder=embeddings.TfidfEmbedder())
    assert result["retrieved"], "no chunks retrieved"
    assert all("score" in c and "doc_id" in c for c in result["retrieved"])
    assert "Django REST Framework" in result["matched"]


# -- composer ----------------------------------------------------------
def test_no_key_composer_is_extractive(tmp_path):
    jd = "Backend Python intern: Flask, SQL, Git."
    result = pipeline.analyze(jd, db_path=str(tmp_path / "e.db"),
                               embedder=embeddings.TfidfEmbedder(),
                               use_llm=False)
    assert result["mode"] == "extractive"
    assert result["narrative"] is None
    assert result["verdict"]
    assert result["talking_points"], "talking points missing"
    assert all(tp["citation"] for tp in result["talking_points"])


def test_composer_verdict_counts_match(tmp_path):
    jd = "We want Flask and React."
    result = pipeline.analyze(jd, db_path=str(tmp_path / "f.db"),
                               embedder=embeddings.TfidfEmbedder(),
                               use_llm=False)
    assert result["match_count"] == len(result["matched"])
    assert result["total_skills"] == len(result["jd_skills"])
    assert "1 of 2" in result["verdict"]


def test_empty_jd_raises(tmp_path):
    with pytest.raises(ValueError):
        pipeline.analyze("   ", db_path=str(tmp_path / "g.db"),
                          embedder=embeddings.TfidfEmbedder())


# -- CLI ---------------------------------------------------------------
def test_cli_runs_on_sample(tmp_path, capsys):
    from careerlens import cli
    db = tmp_path / "cli.db"
    # Point the pipeline at the temp DB via monkeypatched default is
    # complex; instead run analyse directly and format the report.
    result = pipeline.analyze(
        (ROOT / "samples" / "sample_jd_backend.txt").read_text(),
        db_path=str(db), embedder=embeddings.TfidfEmbedder(),
        use_llm=False)
    report = cli.format_report(result)
    assert "CareerLens" in report
    assert "MATCHED SKILLS" in report
    assert "SKILLS GAPS" in report
    assert "Flask" in report
