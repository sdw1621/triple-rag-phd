"""
Strip parentheses around inline [N] citations.

After convert_inline_citations.py, body text has artifacts like:
    Microsoft GraphRAG ([13] 는      ← orphan opening paren
    Self-RAG ([20] 와                ← orphan opening paren
    선행 논문 ([1])                  ← intact parens around [N]
    ([97]; [101],                    ← multi-citation paren left dangling

This script normalizes all of these to IEEE bare-bracket style:
    Microsoft GraphRAG [13] 는
    Self-RAG [20] 와
    선행 논문 [1]
    [97], [101],
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"

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


def convert(text: str) -> tuple[str, dict[str, int]]:
    stats: dict[str, int] = {}

    def _track(label: str, n: int) -> None:
        if n:
            stats[label] = stats.get(label, 0) + n

    # 1. Fully wrapped: ([N])  →  [N]
    text, n = re.subn(r"\(\s*(\[\d+\])\s*\)", r"\1", text)
    _track("([N]) → [N]", n)

    # 2. Multi-citation paren: ([N]; [M])  →  [N], [M]
    # First handle full forms with closing paren
    text, n = re.subn(
        r"\(\s*(\[\d+\](?:\s*;\s*\[\d+\])+)\s*\)",
        lambda m: re.sub(r"\s*;\s*", ", ", m.group(1)),
        text,
    )
    _track("([N]; [M]) → [N], [M]", n)

    # 3. Orphan opening paren before [N] (closing was eaten earlier):
    #    Model ([N] xxx  →  Model [N] xxx
    # Match "(" followed by zero/more spaces then [N]; remove the "("
    text, n = re.subn(r"\(\s*(\[\d+\])", r"\1", text)
    _track("([N]  →  [N]  (orphan open paren)", n)

    # 4. Orphan closing paren after [N] (rare): [N])  →  [N]
    # Only when not preceded by matching "(" within reasonable distance.
    # Simpler: if line has a [N]) where the corresponding ( is far away,
    # we leave it. But for the common case "[N])", strip it.
    # Skip this heuristic — risk of breaking legitimate text.

    # 5. Semicolon between brackets → comma (IEEE convention)
    text, n = re.subn(r"(\[\d+\])\s*;\s*(\[\d+\])", r"\1, \2", text)
    _track("[N];[M] → [N], [M]", n)

    return text, stats


def main() -> int:
    total: dict[str, int] = {}
    files_changed = 0
    for path in KOR_FILES:
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        new_text, stats = convert(original)
        if new_text != original:
            path.write_text(new_text, encoding="utf-8")
            files_changed += 1
            count = sum(stats.values())
            print(f"  ✓ {path.relative_to(ROOT)}: {count} fixes {stats}")
            for k, v in stats.items():
                total[k] = total.get(k, 0) + v
    print()
    print(f"Files changed: {files_changed}")
    print(f"Total fixes:")
    for k, v in sorted(total.items(), key=lambda x: -x[1]):
        print(f"  {v:3d}  {k}")
    print(f"  ----")
    print(f"  {sum(total.values()):3d}  TOTAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
