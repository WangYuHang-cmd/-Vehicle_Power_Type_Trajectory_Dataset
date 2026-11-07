"""Dataset-card rendering: docs/README_template.md + index/readme_numbers.json -> release/README.md.

The template carries `{{placeholder}}` tokens and no digits of its own except the literals listed in
TEMPLATE_LITERALS (license strings, the arXiv id and the bibtex year); every number on the card comes from the
numbers file, every table of columns from vtt.schema / vtt.gate.rules. `render` refuses missing or leftover
placeholders. Public wording: 'driving automation system (DAS)'.
"""
from __future__ import annotations

import re
from pathlib import Path

from vtt import config as C
from vtt import schema as S

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "docs" / "README_template.md"
LICENSE_TEXT_PATH = Path(__file__).resolve().parents[1] / "docs" / "LICENSE_CC_BY_4.0.txt"
# literal strings of the template that may contain digits (removed before the no-digit test)
TEMPLATE_LITERALS = ("cc-by-4.0", "CC BY 4.0", "licenses/by/4.0", "arXiv:2603.06986", "arxiv.org/abs/2603.06986",
                     "@article{wang2026adasto", "year    = {2026}", "windows_60s", "m^2", "sha256", "float32", "0 / 1",
                     "footprint_m2", "macro-averaged F1")
PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


def template_text(path: Path = TEMPLATE_PATH) -> str:
    return Path(path).read_text()


def placeholders(text: str) -> list[str]:
    """Distinct placeholder names in order of first appearance."""
    seen: list[str] = []
    for m in PLACEHOLDER.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def template_digits(text: str) -> list[str]:
    """Digit runs left in the template after removing placeholders and the allowed literals (must be empty)."""
    t = PLACEHOLDER.sub(" ", text)
    for lit in TEMPLATE_LITERALS:
        t = t.replace(lit, " ")
    return re.findall(r"\d+", t)


def fmt(value) -> str:
    """Display format: ints with thousands separators, floats with one decimal, bools yes/no, None 'n/a'."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:,.1f}"
    return str(value)


def render(text: str, numbers: dict) -> str:
    """Substitute every placeholder from `numbers` (KeyError on a missing key, ValueError on leftovers)."""
    missing = [k for k in placeholders(text) if k not in numbers]
    if missing:
        raise KeyError(f"readme numbers missing: {missing}")
    out = PLACEHOLDER.sub(lambda m: fmt(numbers[m.group(1)]), text)
    if "{{" in out or "}}" in out:
        raise ValueError("unrendered placeholder braces left in the README")
    return out


# ------------------------------------------------------------------------------------------------ generated blocks
def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def markdown_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _unit(u: str) -> str:
    return u if u else "-"


def field_table(spec=S.DRIVE_COLUMNS) -> str:
    """Drive-table columns from vtt.schema (name, type, unit, description, upstream field)."""
    rows = []
    for name, dt, unit, desc, origin in spec:
        tag = "input" if name in C.OFFICIAL_INPUTS else "metadata"
        rows.append([f"`{name}`", dt, _unit(unit), desc, origin, tag])
    return markdown_table(["column", "type", "unit", "description", "source field", "role"], rows)


def index_table(spec) -> str:
    """Index-table columns (name, type, description) for drives / windows / vehicles / platforms."""
    return markdown_table(["column", "type", "description"], [[f"`{n}`", dt, d or "-"] for n, dt, d in spec])


def official_inputs_table() -> str:
    units = {t[0]: (t[2], t[3]) for t in S.DRIVE_COLUMNS}
    rows = [[f"`{n}`", _unit(units[n][0]), units[n][1]] for n in C.OFFICIAL_INPUTS]
    return markdown_table(["official input", "unit", "meaning"], rows)


def nuisance_columns_list() -> str:
    """Drive-table columns that are NOT model inputs under the official protocol."""
    names = [t[0] for t in S.DRIVE_COLUMNS if t[0] not in C.OFFICIAL_INPUTS and t[0] != "t_s"]
    return ", ".join(f"`{n}`" for n in names)


def blocks() -> dict[str, str]:
    """All generated README blocks (schema-derived; no numbers from the data)."""
    from vtt.gate import rules as R
    return {
        "field_table": field_table(),
        "official_inputs": official_inputs_table(),
        "nuisance_columns": nuisance_columns_list(),
        "drives_index_table": index_table(S.DRIVES_INDEX_COLUMNS),
        "windows_index_table": index_table(S.WINDOWS_INDEX_COLUMNS),
        "vehicles_table": index_table(R.VEHICLES_COLUMNS),
        "platforms_table": index_table(R.PLATFORMS_COLUMNS),
    }


def rows_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Markdown table from a list of dict rows and (key, header) pairs, values through `fmt`."""
    return markdown_table([h for _, h in columns], [[fmt(r.get(k)) for k, _ in columns] for r in rows])


def license_text(dataset_name: str, url: str) -> str:
    """Release LICENSE: a short notice followed by the CC BY 4.0 legal code."""
    legal = LICENSE_TEXT_PATH.read_text()
    head = (f"{dataset_name}\n\nThe data tables of this release are licensed under the Creative Commons Attribution 4.0\n"
            f"International License (CC BY 4.0). Dataset page: {url}\n"
            "To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/\n\n"
            "The legal code follows.\n\n" + "=" * 72 + "\n\n")
    return head + legal
