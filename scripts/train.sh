#!/bin/sh
cd "$(dirname "$0")/.."
PYTHONUNBUFFERED=1 exec .venv/bin/python -m triage.cli train "$@"
