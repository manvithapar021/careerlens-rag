"""CareerLens — a RAG-powered job-description matcher.

Paste a job description. CareerLens retrieves evidence from a corpus built
from Manvi Thapar's own resume and her GitHub projects' READMEs, then
produces a grounded fit analysis with citations, talking points, and an
honest skills-gap list. It never invents experience she does not have:
any skill in the JD with no supporting evidence is listed as a gap,
never claimed.
"""

__version__ = "1.0.0"
