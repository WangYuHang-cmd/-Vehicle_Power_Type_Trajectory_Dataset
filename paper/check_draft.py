#!/usr/bin/env python3
"""Draft checks for the paper sources: forbidden wording, identifiers, non-ASCII bytes, citation keys, labels, numbers.
Usage: python3 check_draft.py [paper_dir]. Exit code 1 when a hard finding exists."""
import re, sys, pathlib

P = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent)
tex = sorted((P / "sections").glob("*.tex")) + [P / "main.tex"]
bib = (P / "refs.bib").read_text()
keys = set(re.findall(r"@\w+\{([^,\s]+),", bib)) - {"IEEEexample:BSTcontrol"}
hard, soft = [], []
forbidden = re.compile(r"\bA\s*D\s*A\s*S\b(?!-TO)", re.I)
for f in tex:
    s = f.read_text(errors="replace")
    for i, line in enumerate(s.splitlines(), 1):
        body = line.split("%", 1)[0] if not line.lstrip().startswith("%") else ""
        if forbidden.search(body):
            hard.append(f"{f.name}:{i}: forbidden driver-assistance acronym: {line.strip()[:100]}")
        if re.search(r"[^\x00-\x7f]", line):
            hard.append(f"{f.name}:{i}: non-ASCII: {line.strip()[:100]}")
        if re.search(r"\b[0-9a-f]{16}\b", body) or re.search(r"\d{4}-\d{2}-\d{2}", body) or re.search(r"/(?:home|data|mnt|tmp)/", body):
            hard.append(f"{f.name}:{i}: identifier / date / path: {line.strip()[:100]}")
        if re.search(r"[\w.]+@[\w.]+\.\w+", body):
            hard.append(f"{f.name}:{i}: e-mail: {line.strip()[:100]}")
        for k in re.findall(r"\\cite[tp]?\{([^}]*)\}", body):
            for kk in k.split(","):
                if kk.strip() and kk.strip() not in keys:
                    hard.append(f"{f.name}:{i}: unknown citation key {kk.strip()}")
        for m in re.findall(r"\\(si|num|SI|todo|cref|Cref)\b", body):
            hard.append(f"{f.name}:{i}: forbidden macro \\{m}")
all_tex = "\n".join(f.read_text(errors="replace") for f in tex)
labels = set(re.findall(r"\\label\{([^}]+)\}", all_tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", all_tex))
for r in sorted(refs - labels):
    hard.append(f"undefined reference {r}")
for l in sorted(labels - refs):
    soft.append(f"label never referenced {l}")
dup = [l for l in labels if all_tex.count("\\label{" + l + "}") > 1]
for l in dup:
    hard.append(f"duplicate label {l}")
todos = re.findall(r"%\s*TODO[^\n]*", all_tex)
words = len(re.sub(r"%[^\n]*", "", re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", all_tex)).split())
print(f"files {len(tex)}  words(approx) {words}  labels {len(labels)}  todos {len(todos)}")
for t in todos:
    print("  ", t.strip()[:120])
print(f"hard findings {len(hard)}")
for h in hard:
    print("  HARD", h)
print(f"soft findings {len(soft)}")
for s_ in soft:
    print("  soft", s_)
sys.exit(1 if hard else 0)
