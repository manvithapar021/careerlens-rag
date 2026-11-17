"""Answer composition: grounded, cited, and honest.

Two composers behind one interface:

- ``compose_extractive`` — works with NO API key. Builds the fit
  analysis from the retrieved chunks with templates: matched skills
  with citations, gaps, talking points, and a verdict. This is the
  default mode and the demo runs entirely on it.
- ``compose_with_llm`` — when ``CAREERLENS_LLM_PROVIDER`` and the
  matching API key env var are set (OpenAI / Groq / Gemini), a hosted
  LLM writes the narrative section from the same retrieved evidence,
  under a strict system prompt: only claim skills with cited evidence,
  list everything else as a gap. The deterministic matched/gap lists
  are NEVER overridden by the LLM — they are computed by
  ``skills.classify_skills`` and shown alongside.

No mode ever implies the no-key path is calling a hosted LLM.
"""

import os

import requests

LLM_PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "url": None,  # built per-model below
        "default_model": "gemini-2.0-flash",
    },
}

SYSTEM_PROMPT = (
    "You are a career coach helping a student analyse a job description "
    "against her own experience. You are given retrieved evidence chunks "
    "from her resume and project READMEs, each with a citation id like "
    "[resume] or [safecampus]. Rules: 1) Only claim a skill if a cited "
    "chunk supports it. 2) Any skill in the job description with no "
    "supporting chunk must be listed as a gap, never claimed. 3) Cite "
    "every factual claim with its chunk id. 4) Be concise and specific: "
    "name the project and what was built. 5) Do not invent experience, "
    "numbers, or employers."
)


class ComposeError(Exception):
    """Raised when the hosted-LLM path fails (caller falls back)."""


def llm_configured():
    """(provider, api_key, model) if an LLM is configured, else None."""
    provider = os.environ.get("CAREERLENS_LLM_PROVIDER", "").lower().strip()
    if provider not in LLM_PROVIDERS:
        return None
    key = os.environ.get(LLM_PROVIDERS[provider]["env_key"], "").strip()
    if not key:
        return None
    model = os.environ.get("CAREERLENS_MODEL") or \
        LLM_PROVIDERS[provider]["default_model"]
    return provider, key, model


def _call_openai_compatible(url, api_key, model, system, user):
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model, "temperature": 0.3,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(api_key, model, system, user):
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": system + "\n\n" + user}]}]},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def compose_with_llm(jd_text, retrieved):
    """Ask the configured hosted LLM for a grounded narrative.

    Raises ComposeError on any failure so the caller can fall back to
    the extractive composer.
    """
    cfg = llm_configured()
    if cfg is None:
        raise ComposeError("No LLM provider configured")
    provider, key, model = cfg
    evidence = "\n\n".join(
        f"[{c['doc_id']}] {c['title']} (score {c['score']:.2f}): {c['text']}"
        for c in retrieved
    )
    user = (f"JOB DESCRIPTION:\n{jd_text}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence}\n\n"
            "Write a short fit analysis: 1) a 2-3 sentence verdict, "
            "2) matched skills with citations, 3) gaps with citations "
            "marked as not evidenced, 4) two talking points for the "
            "interview, each citing the project it comes from.")
    try:
        if provider == "gemini":
            narrative = _call_gemini(key, model, SYSTEM_PROMPT, user)
        else:
            url = LLM_PROVIDERS[provider]["url"]
            narrative = _call_openai_compatible(
                url, key, model, SYSTEM_PROMPT, user)
    except Exception as exc:  # network, auth, quota, bad response...
        raise ComposeError(f"{provider} call failed: {exc}") from exc
    return {"mode": f"llm:{provider}/{model}", "narrative": narrative}


def compose_extractive(jd_text, matched, gaps, retrieved, jd_skills):
    """No-key composer: templates over retrieved evidence.

    Every claim traces to a retrieved chunk; nothing is invented.
    """
    total = len(jd_skills)
    n_matched = len(matched)
    if total == 0:
        verdict = ("No recognised skills were found in this job "
                   "description, so no fit score can be computed. The "
                   "retrieved projects below are the closest matches by "
                   "text similarity.")
    elif n_matched == total:
        verdict = (f"Strong fit: all {total} skills named in this job "
                   "description are evidenced by Manvi's resume and "
                   "projects.")
    elif n_matched == 0:
        verdict = (f"No direct fit: none of the {total} skills named in "
                   "this job description are evidenced by Manvi's resume "
                   "and projects. See the gap list before applying.")
    else:
        verdict = (f"Partial fit: {n_matched} of {total} skills named in "
                   f"this job description are evidenced by Manvi's resume "
                   f"and projects; {total - n_matched} are gaps.")

    talking_points = []
    for chunk in retrieved[:3]:
        talking_points.append({
            "text": (f"{chunk['title']}: "
                     f"{chunk['text'][:220].strip()}"
                     f"{'...' if len(chunk['text']) > 220 else ''}"),
            "citation": chunk["doc_id"],
            "source": chunk["source"],
            "score": chunk["score"],
        })

    return {
        "mode": "extractive",
        "verdict": verdict,
        "match_count": n_matched,
        "total_skills": total,
        "matched": matched,
        "gaps": gaps,
        "talking_points": talking_points,
        "narrative": None,
    }
