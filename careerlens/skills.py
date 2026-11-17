"""Skill vocabulary, JD skill extraction, and honest gap detection.

Two ideas make this module the heart of CareerLens' honesty rule:

1. ``SKILL_PATTERNS`` maps every skill a job description might ask for
   to the regexes that detect it. It covers Manvi's real skills AND the
   skills she does *not* have, so gaps can be named instead of ignored.
2. ``HER_SKILLS`` is the closed set of skills she can actually evidence
   from her resume and projects. A JD skill is "matched" only when it is
   in ``HER_SKILLS`` *and* supporting text exists in the corpus. Anything
   else is a gap — never claimed.
"""

import re

# Canonical skill -> regex patterns that detect a mention of it.
# Patterns are matched case-insensitively with word boundaries where it
# matters (so "Java" never matches "JavaScript", and "C" never matches
# "C#" or "C++").
SKILL_PATTERNS = {
    # --- Manvi's real skills ---
    "Python": [r"\bpython\b"],
    "SQL": [r"\bsql\b"],
    "C": [r"(?<!\w)C(?![\w+#])"],
    "C++": [r"C\+\+"],
    "Java": [r"\bjava\b(?!\w)"],
    "C#": [r"C#"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "Flask": [r"\bflask\b"],
    "Django": [r"\bdjango\b"],
    "Django REST Framework": [r"django rest framework", r"\bdrf\b"],
    "SQLAlchemy": [r"\bsqlalchemy\b"],
    "Flask-Login": [r"flask-login"],
    "Flask-Mail": [r"flask-mail"],
    "HTML": [r"\bhtml\b", r"\bhtml5\b"],
    "CSS": [r"\bcss\b", r"\bcss3\b"],
    "Bootstrap": [r"\bbootstrap\b"],
    "Leaflet.js": [r"leaflet"],
    "Pandas": [r"\bpandas\b"],
    "Matplotlib": [r"\bmatplotlib\b"],
    "MS Excel": [r"\bexcel\b", r"\bms excel\b", r"microsoft excel"],
    "Power BI": [r"power bi"],
    "Tableau": [r"\btableau\b"],
    "Git": [r"\bgit\b"],
    "GitHub": [r"\bgithub\b"],
    "Railway": [r"\brailway\b"],
    "REST APIs": [r"rest api", r"restful api", r"\brest\b"],
    "CRUD": [r"\bcrud\b"],
    "OOP": [r"\boop\b", r"object[- ]oriented programming"],
    "Data Structures & Algorithms": [
        r"data structures", r"\bdsa\b", r"\balgorithms\b"],
    "DBMS": [r"\bdbms\b", r"database management"],
    "Operating Systems": [r"operating systems", r"\bos\b"],
    "GenAI": [r"genai", r"generative ai"],
    "LLM": [r"\bllm\b", r"large language model"],
    "RAG": [r"\brag\b", r"retrieval[- ]augmented generation"],
    "Prompt Engineering": [r"prompt engineering", r"prompting"],
    "Embeddings": [r"\bembeddings?\b"],
    "Vector Databases": [r"vector (database|store|search)", r"\bfaiss\b"],
    # --- Skills she does NOT claim (gap detection) ---
    "React": [r"\breact\b", r"react\.js"],
    "Node.js": [r"node\.?js", r"\bnode\b"],
    "TypeScript": [r"\btypescript\b"],
    "Angular": [r"\bangular\b"],
    "Vue.js": [r"\bvue\b"],
    "Next.js": [r"next\.js"],
    "FastAPI": [r"\bfastapi\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Scikit-learn": [r"scikit-learn", r"\bsklearn\b"],
    "Machine Learning": [r"machine learning", r"\bml\b"],
    "AWS": [r"\baws\b", r"amazon web services"],
    "Docker": [r"\bdocker\b"],
    "Kubernetes": [r"\bkubernetes\b"],
    "Spark": [r"\bspark\b", r"\bpyspark\b"],
    "Airflow": [r"\bairflow\b"],
    "Kafka": [r"\bkafka\b"],
    "MongoDB": [r"\bmongodb\b", r"\bmongo\b"],
    "PostgreSQL": [r"\bpostgres", r"\bpostgresql\b"],
    "MySQL": [r"\bmysql\b"],
    "Redis": [r"\bredis\b"],
    "GraphQL": [r"\bgraphql\b"],
    "Go": [r"\bgo\b", r"\bgolang\b"],
    "Rust": [r"\brust\b"],
    "Kotlin": [r"\bkotlin\b"],
    "Flutter": [r"\bflutter\b"],
    "Android": [r"\bandroid\b"],
    "Azure": [r"\bazure\b"],
    "GCP": [r"\bgcp\b", r"google cloud"],
    "CI/CD": [r"\bci/cd\b", r"continuous integration"],
    "Terraform": [r"\bterraform\b"],
    "Databricks": [r"\bdatabricks\b"],
    "LangChain": [r"\blangchain\b"],
}

_COMPILED = {
    skill: [re.compile(p, re.IGNORECASE) for p in patterns]
    for skill, patterns in SKILL_PATTERNS.items()
}

# The closed set of skills Manvi can evidence from her resume/projects.
HER_SKILLS = {
    "Python", "SQL", "C", "C++", "Java", "C#", "JavaScript",
    "Flask", "Django", "Django REST Framework", "SQLAlchemy",
    "Flask-Login", "Flask-Mail", "HTML", "CSS", "Bootstrap", "Leaflet.js",
    "Pandas", "Matplotlib", "MS Excel", "Power BI", "Tableau",
    "Git", "GitHub", "Railway", "REST APIs", "CRUD", "OOP",
    "Data Structures & Algorithms", "DBMS", "Operating Systems",
    "GenAI", "LLM", "RAG", "Prompt Engineering", "Embeddings",
    "Vector Databases",
}


def extract_skills(text):
    """All vocabulary skills mentioned in ``text``, in order of mention."""
    found = []
    for skill, patterns in _COMPILED.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                found.append((m.start(), skill))
                break
    found.sort()
    return [skill for _, skill in found]


def skill_in_text(skill, text):
    """True when any of the skill's patterns matches ``text``."""
    return any(pat.search(text) for pat in _COMPILED[skill])


def classify_skills(jd_skills, evidence_chunks):
    """Split JD skills into matched (evidenced) and gaps.

    ``evidence_chunks`` is the list of corpus chunks (dicts with a
    "text" field). A skill counts as matched only when it is one of
    Manvi's real skills AND at least one chunk evidences it. Everything
    else is a gap — the "refuse to claim" rule.
    """
    matched, gaps = [], []
    for skill in jd_skills:
        evidenced = any(
            skill_in_text(skill, chunk["text"]) for chunk in evidence_chunks
        )
        if skill in HER_SKILLS and evidenced:
            matched.append(skill)
        else:
            gaps.append(skill)
    return matched, gaps


def supporting_chunks(skill, chunks):
    """Chunks that evidence ``skill``, best first (caller orders)."""
    return [c for c in chunks if skill_in_text(skill, c["text"])]
