"""
Reorder references by first citation order (IEEE strict style).

Steps:
  1. Walk body files in document order, collect [N] in order of first
     appearance. This becomes the new sequence.
  2. Build mapping old_num → new_num:
       - cited refs: positions 1..K (K = unique cited count)
       - uncited refs: positions K+1..100, preserving their old order
  3. Apply mapping to:
       - 09_참고문헌.md: renumber entries AND reorder them so the file lists
         them in the new sequence
       - body files: renumber inline [N] citations

After this, body shows [1], [2], [3], ... sequentially as the reader
encounters new references.

Note: category headers (## [1] Prior Work, ## [2] RAG Foundations, ...)
in the references file no longer correspond to a contiguous range. They
are removed, and the file becomes a flat 100-entry IEEE list.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"

# Files in document order (matches build_unified_thesis_v25.py ORDER, body part)
BODY_FILES_IN_ORDER = [
    V20 / "01_서론.md",
    V20 / "02_관련연구.md",
    V20 / "03_아키텍처.md",
    V20 / "04_R-DWA.md",
    V20 / "05_L-DWA.md",
    V20 / "06_실험평가.md",
    V20 / "06a_CS1-6_case_studies.md",
    V20 / "06b_CS7_list_prompt.md",
    V20 / "07_결론.md",
    V20 / "08_부록.md",
    V5 / "00b_국문초록.md",
    V5 / "10_감사의글.md",
]

REFERENCES = V5 / "09_참고문헌.md"


def extract_citation_order() -> list[int]:
    """Return list of unique [N] in first-encounter order across body files."""
    seen: set[int] = set()
    order: list[int] = []
    bracket_re = re.compile(r"\[(\d+)\]")
    for path in BODY_FILES_IN_ORDER:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for m in bracket_re.finditer(text):
            n = int(m.group(1))
            if n not in seen:
                seen.add(n)
                order.append(n)
    return order


def parse_references(text: str) -> tuple[list[str], list[tuple[int, str]]]:
    """Split references file into (header_lines, entries).

    Each entry is a (number, full_block_text) tuple where full_block_text
    includes the entry [N] X. Y..." text and any blank lines until the
    next [M] entry. Category headers like '## [1] Prior Work' are dropped.
    """
    lines = text.splitlines(keepends=True)
    # Find the first [N] entry — everything before that is the file header
    entry_start_re = re.compile(r"^\[(\d+)\]\s")
    category_header_re = re.compile(r"^##\s+\[\d+\]\s")
    header_end = 0
    for i, ln in enumerate(lines):
        if entry_start_re.match(ln):
            header_end = i
            break
    else:
        raise RuntimeError("No [N] entry found in references")

    header_lines = lines[:header_end]
    # Drop category headers from the header part if any (they shouldn't be
    # there but be safe). Also drop them from the body part.
    body_lines = [ln for ln in lines[header_end:] if not category_header_re.match(ln)]

    # Parse body_lines into entries
    entries: list[tuple[int, list[str]]] = []
    current_num = None
    current_block: list[str] = []
    for ln in body_lines:
        m = entry_start_re.match(ln)
        if m:
            if current_num is not None:
                entries.append((current_num, current_block))
            current_num = int(m.group(1))
            current_block = [ln]
        else:
            # Stop collecting if we hit a section break like '---' followed by
            # editorial conventions (after the last entry). Detect by HR rule.
            if ln.strip() == "---" and current_num is not None:
                # Save current entry, then collect the rest as trailing
                entries.append((current_num, current_block))
                current_num = None
                current_block = []
                # Append the rest as trailing footer
                idx = body_lines.index(ln)
                trailing = body_lines[idx:]
                # Reassign trailing as a sentinel
                return header_lines, entries, trailing
            if current_num is not None:
                current_block.append(ln)
    if current_num is not None:
        entries.append((current_num, current_block))
    return header_lines, entries, []


def main() -> int:
    cited_order = extract_citation_order()
    print(f"Found {len(cited_order)} unique citations in body, first-encounter order:")
    print(f"  {cited_order}")

    # Build mapping
    cited_set = set(cited_order)
    all_refs_in_order: list[int] = list(cited_order)
    # Append uncited refs in their old numeric order
    for n in range(1, 101):
        if n not in cited_set:
            all_refs_in_order.append(n)
    assert len(all_refs_in_order) == 100, f"Expected 100, got {len(all_refs_in_order)}"

    # Mapping: old_num → new_num (1-indexed position in all_refs_in_order)
    remap: dict[int, int] = {old: new + 1 for new, old in enumerate(all_refs_in_order)}
    print(f"Mapping (cited refs in body-encounter order):")
    for old in cited_order:
        print(f"  [{old}] → [{remap[old]}]")

    # Apply mapping to body files (single-pass)
    body_total = 0

    def _shift(text: str) -> tuple[str, int]:
        counter = {"n": 0}

        def _sub(m: re.Match) -> str:
            n = int(m.group(1))
            if n in remap:
                counter["n"] += 1
                return f"[{remap[n]}]"
            return m.group(0)

        return re.sub(r"\[(\d+)\]", _sub, text), counter["n"]

    for path in BODY_FILES_IN_ORDER:
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        new_text, n = _shift(original)
        if new_text != original:
            path.write_text(new_text, encoding="utf-8")
            body_total += n
            print(f"  ✓ {path.name}: {n} citations renumbered")

    print(f"Body total: {body_total}")

    # Reorder + renumber references file
    ref_text = REFERENCES.read_text(encoding="utf-8")
    header_lines, entries, trailing = parse_references(ref_text)

    # Convert entries to dict for quick lookup
    entries_by_num: dict[int, list[str]] = {n: lines for n, lines in entries}

    # Build new references body in remapped order
    new_body_lines: list[str] = []
    for new_pos, old_num in enumerate(all_refs_in_order, start=1):
        block = entries_by_num.get(old_num)
        if block is None:
            print(f"WARN: missing entry for old [{old_num}]")
            continue
        # Renumber the leading "[old_num]" → "[new_pos]"
        first = block[0]
        first_renum = re.sub(rf"^\[{old_num}\]", f"[{new_pos}]", first, count=1)
        new_block = [first_renum] + block[1:]
        new_body_lines.extend(new_block)
        # Add a blank line between entries if not present
        if new_block and not new_block[-1].endswith("\n\n"):
            new_body_lines.append("\n")

    # Also remap any cross-references inside the trailing/footer text
    # (e.g., "entry [32]" appearing in editorial conventions).
    trailing_text = "".join(trailing)

    def _sub_cross(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[{remap.get(n, n)}]"

    trailing_text = re.sub(r"\[(\d+)\]", _sub_cross, trailing_text)

    new_text = "".join(header_lines) + "".join(new_body_lines) + trailing_text
    REFERENCES.write_text(new_text, encoding="utf-8")
    print(f"  ✓ {REFERENCES.name}: 100 entries reordered and renumbered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
