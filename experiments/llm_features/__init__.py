"""Physics-based window features proposed by LLM agents (see experiments/LLM_FEATURES.md).

Every module f_<name>.py holds one feature function; FEATURES maps the feature name to it. Built by
experiments/llm_features_build.py from the verified proposals; do not edit the feature modules by hand.
"""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

MANIFEST = json.loads((Path(__file__).parent / '_manifest.json').read_text())
FEATURES = {}
for _m in MANIFEST:
    if _m['status'] != 'kept':
        continue
    _mod = import_module(f"experiments.llm_features.f_{_m['name']}")
    FEATURES[_m['name']] = getattr(_mod, _m['function'])

NAMES = list(FEATURES)
