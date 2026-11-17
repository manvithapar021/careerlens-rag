"""CareerLens CLI: analyse a job description from the terminal.

Usage:
    python -m careerlens.cli --jd samples/sample_jd_backend.txt
    python -m careerlens.cli --jd samples/sample_jd_ai_ml.txt --top-k 5
    python -m careerlens.cli --jd jd.txt --llm        # hosted LLM if configured
    python -m careerlens.cli --rebuild                # rebuild the index first
"""

import argparse
import sys
from pathlib import Path

from . import pipeline


def format_report(result):
    lines = []
    lines.append("=" * 64)
    lines.append("CareerLens — fit analysis")
    lines.append(f"Mode: {result['mode']} | Embedder: {result['embedder']}")
    lines.append("=" * 64)
    lines.append("")
    lines.append(result["verdict"])
    lines.append("")
    if result.get("narrative"):
        lines.append("--- LLM analysis ---")
        lines.append(result["narrative"])
        lines.append("")
    if result.get("llm_error"):
        lines.append(f"(LLM call failed: {result['llm_error']}; "
                      "showing extractive analysis.)")
        lines.append("")
    lines.append(f"MATCHED SKILLS ({result['match_count']}/"
                 f"{result['total_skills']}):")
    for skill in result["matched"]:
        cit = result["citations"].get(skill, {})
        score = cit.get("score")
        score_txt = f" (similarity {score:.2f})" if score is not None else ""
        lines.append(f"  [x] {skill} — {cit.get('title', '?')}"
                     f" [{cit.get('doc_id', '?')}] {score_txt}")
    lines.append("")
    lines.append("SKILLS GAPS (not claimed — no supporting evidence):")
    for skill in result["gaps"]:
        lines.append(f"  [ ] {skill}")
    if not result["gaps"]:
        lines.append("  (none)")
    lines.append("")
    lines.append("TALKING POINTS:")
    for tp in result["talking_points"]:
        lines.append(f"  - {tp['text']} [{tp['citation']}]")
    lines.append("")
    lines.append("RETRIEVED EVIDENCE:")
    for chunk in result["retrieved"]:
        lines.append(f"  [{chunk['doc_id']}] {chunk['title']} — "
                     f"similarity {chunk['score']:.2f}")
        lines.append(f"      {chunk['text'][:180]}...")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jd", help="Path to a text file with the job description")
    parser.add_argument("--jd-text", help="Job description as a string")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", action="store_true",
                        help="Use the hosted LLM if one is configured")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the vector index before analysing")
    args = parser.parse_args(argv)

    if args.rebuild:
        n = pipeline.build_index()
        print(f"Index rebuilt: {n} chunks.")

    if args.jd:
        jd_text = Path(args.jd).read_text(encoding="utf-8")
    elif args.jd_text:
        jd_text = args.jd_text
    else:
        parser.error("Provide --jd <file> or --jd-text <text>")

    result = pipeline.analyze(jd_text, top_k=args.top_k,
                               use_llm=args.llm if args.llm else None)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
