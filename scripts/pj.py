"""Small helper used during the build to update project.json (the dashboard entry).

Usage: python scripts/pj.py patch.json   -> shallow-merges the patch; for "phases" it merges
by phase name, and for list fields prefixed with "+" it appends.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
target = ROOT / "project.json"
doc = json.loads(target.read_text()) if target.exists() else {}
patch = json.loads(Path(sys.argv[1]).read_text())
for key, value in patch.items():
    if key == "phases" and "phases" in doc:
        by_name = {p["name"]: p for p in doc["phases"]}
        for phase in value:
            by_name[phase["name"]].update(phase)
    elif key.startswith("+"):
        doc.setdefault(key[1:], []).extend(value)
    else:
        doc[key] = value
target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
