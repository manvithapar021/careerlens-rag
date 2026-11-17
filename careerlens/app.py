"""CareerLens web app: paste a job description, get a grounded fit analysis.

Routes:
- GET  /            — form + sample job descriptions
- POST /analyze    — HTML result page
- GET  /api/analyze?jd=... / POST JSON {"jd": ...} — JSON result
- GET  /health     — index status (chunk count, embedder, mode)
"""

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from . import pipeline

BASE_DIR = Path(__file__).resolve().parent.parent


def create_app(db_path=None, corpus_dir=None):
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
    )
    app.config["DB_PATH"] = db_path or str(pipeline.DEFAULT_DB_PATH)
    app.config["CORPUS_DIR"] = corpus_dir or str(pipeline.DEFAULT_CORPUS_DIR)

    @app.route("/", methods=["GET"])
    def index():
        samples = {}
        samples_dir = BASE_DIR / "samples"
        for path in sorted(samples_dir.glob("*.txt")):
            samples[path.stem] = path.read_text(encoding="utf-8")
        return render_template("index.html", samples=samples)

    @app.route("/analyze", methods=["POST"])
    def analyze_form():
        jd_text = request.form.get("jd", "").strip()
        if not jd_text:
            return render_template(
                "index.html", samples=_samples(), error="Paste a job description first.")
        try:
            result = pipeline.analyze(jd_text, db_path=app.config["DB_PATH"],
                                       corpus_dir=app.config["CORPUS_DIR"])
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            return render_template("index.html", samples=_samples(),
                                   error=f"Analysis failed: {exc}")
        return render_template("result.html", jd=jd_text, result=result)

    @app.route("/api/analyze", methods=["GET", "POST"])
    def analyze_api():
        if request.method == "POST":
            jd_text = (request.get_json(silent=True) or {}).get("jd", "")
        else:
            jd_text = request.args.get("jd", "")
        if not jd_text.strip():
            return jsonify({"error": "Provide a job description as 'jd'."}), 400
        result = pipeline.analyze(jd_text, db_path=app.config["DB_PATH"],
                                   corpus_dir=app.config["CORPUS_DIR"])
        return jsonify(result)

    @app.route("/health", methods=["GET"])
    def health():
        from .store import VectorStore
        store = VectorStore(app.config["DB_PATH"])
        try:
            return jsonify({
                "status": "ok",
                "chunks": store.count(),
                "embedder": store.get_meta("embedder"),
            })
        finally:
            store.close()

    def _samples():
        samples = {}
        for path in sorted((BASE_DIR / "samples").glob("*.txt")):
            samples[path.stem] = path.read_text(encoding="utf-8")
        return samples

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
