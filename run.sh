#!/bin/sh
# CareerLens quick start: create venv, install deps, run the web app.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python build_index.py
.venv/bin/python -m careerlens.cli --jd samples/sample_jd_backend.txt
echo ""
echo "Starting the web app at http://127.0.0.1:5000 ..."
.venv/bin/python -m careerlens.app
