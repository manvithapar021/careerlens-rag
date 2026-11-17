# CareerLens — RAG Job-Description Matcher

CareerLens is a retrieval-augmented generation application built with
Python and Flask. A user pastes a job description, and CareerLens
retrieves evidence from a corpus of the candidate's resume and
project READMEs, then produces a grounded fit analysis with citations,
talking points and an honest skills-gap list.

The RAG pipeline is document ingestion and chunking, local embeddings,
a SQLite-backed vector store, top-k retrieval with cosine similarity
scores, and a grounded answer composer. Embeddings are pluggable:
sentence-transformers dense embeddings when available, with a
from-scratch TF-IDF embedder as the fallback, so the demo runs with no
API key and no downloads.

The language model layer is pluggable through environment variables
and supports OpenAI, Groq and Gemini. With no key set, CareerLens
uses an extractive template composer over the retrieved chunks, and
the README states clearly that this mode does not call a hosted LLM.

A strict honesty rule is built in: any skill in the job description
with no supporting evidence in the corpus is listed as a gap and is
never claimed. The project has automated tests covering chunking,
retrieval relevance, citations, gap detection and the no-key composer.
