"""Chunking: split documents into overlapping word-window chunks.

Each chunk carries its document id, title and source so retrieved chunks
can be cited back to the resume or the project they came from.
"""

DEFAULT_CHUNK_WORDS = 120
DEFAULT_OVERLAP_WORDS = 30


def chunk_text(text, chunk_words=DEFAULT_CHUNK_WORDS,
               overlap_words=DEFAULT_OVERLAP_WORDS):
    """Split ``text`` into overlapping word windows.

    Returns a list of chunk strings. Adjacent chunks share
    ``overlap_words`` words so a fact straddling a boundary is not lost.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        chunks.append(" ".join(part))
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_document(doc_id, title, source, text,
                   chunk_words=DEFAULT_CHUNK_WORDS,
                   overlap_words=DEFAULT_OVERLAP_WORDS):
    """Chunk one corpus document into a list of chunk dicts.

    The document title is prepended to every chunk's indexed text so
    retrieval always knows which project a chunk belongs to.
    """
    out = []
    for i, chunk in enumerate(chunk_text(text, chunk_words, overlap_words)):
        out.append({
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "chunk_index": i,
            "text": chunk,
            # Indexed text carries the title for better retrieval;
            # display text stays clean.
            "indexed_text": f"{title}. {chunk}",
        })
    return out
