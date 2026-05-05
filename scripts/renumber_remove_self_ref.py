"""
Remove [2] (this dissertation self-reference) and renumber all entries [3]–[101]
down by one to [2]–[100]. Apply both in the references file and inline body
citations.

Standard academic practice: a dissertation should not list itself as a
reference entry. Removing [2] gives a clean 100-entry list.

Steps:
  1. Read references file, drop [2] block (entry [2] D.-W. Shin, this dissertation)
  2. Rewrite all [N] occurrences with N>=3 to [N-1] in the references file
     (both the entry header "[N]" and any internal references like "entry [32]")
  3. Apply same [N] → [N-1] (N>=3) shift to all body files where inline
     citations exist.

Order matters: shift FROM HIGHEST to LOWEST so [3]→[2] doesn't conflict with
the new [3] coming from [4]. So we go [101]→[100], [100]→[99], …, [3]→[2].
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"

REFERENCES = V5 / "09_참고문헌.md"

KOR_FILES = [
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


def remove_entry_2_from_references(text: str) -> str:
    """Remove the [2] D.-W. Shin (this dissertation) entry from references."""
    # Pattern: from "[2] D.-W. Shin," through to (but not including) the next "[3]"
    # Must be careful: the [2] entry is followed by a blank line then "[3]"
    pattern = re.compile(
        r"\[2\]\s+D\.-W\. Shin,\s+\"Performance Optimization of Triple-Hybrid"
        r".*?\(this dissertation\)\s*\n+",
        re.DOTALL,
    )
    new_text, n = pattern.subn("", text)
    if n != 1:
        raise RuntimeError(f"Expected exactly 1 match for [2] removal, got {n}")
    return new_text


def shift_brackets_down_by_one(text: str, max_n: int = 101) -> tuple[str, int]:
    """Replace [N] with [N-1] for N>=3, in a SINGLE pass.

    Using a multi-pass for-loop is buggy: after [101]→[100], the next pass
    [100]→[99] would also match the freshly-rewritten [100], cascading every
    bracket down to [2]. Single-pass with a callback avoids this.
    """
    counter = {"n": 0}

    def _shift(m: re.Match) -> str:
        n = int(m.group(1))
        if n >= 3:
            counter["n"] += 1
            return f"[{n - 1}]"
        return m.group(0)

    text = re.sub(r"\[(\d+)\]", _shift, text)
    return text, counter["n"]


def main() -> int:
    # 1. References file
    ref_text = REFERENCES.read_text(encoding="utf-8")
    ref_text = remove_entry_2_from_references(ref_text)
    ref_text, ref_count = shift_brackets_down_by_one(ref_text)
    # Update '101 entries' annotation if present
    ref_text = ref_text.replace("101 entries", "100 entries")
    ref_text = ref_text.replace("17 categories, 101", "17 categories, 100")
    REFERENCES.write_text(ref_text, encoding="utf-8")
    print(f"  ✓ {REFERENCES.name}: removed [2], shifted {ref_count} bracket references")

    # 2. Body files
    total_body = 0
    for p in KOR_FILES:
        if not p.exists():
            continue
        original = p.read_text(encoding="utf-8")
        new_text, n = shift_brackets_down_by_one(original)
        if new_text != original:
            p.write_text(new_text, encoding="utf-8")
            total_body += n
            print(f"  ✓ {p.relative_to(ROOT)}: shifted {n} inline citations")

    print()
    print(f"Total bracket shifts in body: {total_body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
