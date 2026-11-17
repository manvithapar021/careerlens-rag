"""Rebuild the CareerLens vector index from data/corpus."""

from careerlens import pipeline

if __name__ == "__main__":
    n = pipeline.build_index()
    print(f"Index built: {n} chunks in data/careerlens.db")
