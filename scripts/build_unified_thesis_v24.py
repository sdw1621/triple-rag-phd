"""
Build the unified PhD thesis v24 — IEEE [N] inline citations applied.

vs v23:
  v23: body uses author-year style ("Lewis et al. (2020)") while references
       file uses IEEE numbered style. Mismatch made citation lookup invisible.
  v24: body inline citations converted to [N] form matching the references
       list. Author names dropped — readers see "[3]의 RAG 기본형은" etc.

  변환된 패턴 (총 22건):
    Lewis et al. (2020)         → [3]
    Asai et al. 2023/2024       → [20]
    Yan et al. 2024             → [21]
    Edge et al. 2024            → [13]
    Jeong et al. (2024)         → [26]
    Schulman et al. (2017)      → [39]
    Schulman et al. (2015)      → [37]
    Yuan et al., 2024           → [52]
    Pineau et al., 2021         → [97]
    Colas et al., 2018          → [101]
    Shin & Moon, 2025, JKSCI    → [1]

References file extended: 100 → 101 entries (added [101] Colas et al. 2018).

vs v22/v21 누적 정정 (모두 v24 에 포함):
  1. semantic Heading 1/2/3/4 styles (commit 0b216f7)
  2. stale → corrected baseline (9313990)
  3. paired bootstrap statistical defense (3e8a4e2)
  4. Ch.1 서론 정합성 (ce41f4a)
  5. References English + author corrections (40c8cfe / 64de61e)
  6. blockquote 내부 markdown 표 fix (f544698)
  7. Ⅰ장 N절 표기 통일 (088a5ce / v23)

Usage:
    python scripts/build_unified_thesis_v24.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"
OUT_MD = V5 / "통합본_v24.md"
OUT_DOCX = V5 / "박사논문_통합본_v24.docx"

ORDER = [
    ("00a_표지.md", None, V5),
    ("01_서론.md", None, V20),
    ("02_관련연구.md", None, V20),
    ("03_아키텍처.md", None, V20),
    ("04_R-DWA.md", None, V20),
    ("05_L-DWA.md", None, V20),
    ("06_실험평가.md", None, V20),
    ("06a_CS1-6_case_studies.md", None, V20),
    ("06b_CS7_list_prompt.md", None, V20),
    ("07_결론.md", None, V20),
    ("08_부록.md", None, V20),
    ("00b_국문초록.md", "국문초록", V5),
    ("00c_영문초록.md", "ABSTRACT", V5),
    ("09_참고문헌.md", "References", V5),
    ("10_감사의글.md", "감사의 글", V5),
]

PAGE_BREAK = "\n\n---\n\n"


def main() -> int:
    buffer: list[str] = []
    missing: list[str] = []

    for fname, override_title, base_dir in ORDER:
        p = base_dir / fname
        if not p.exists():
            missing.append(str(p))
            continue
        text = p.read_text(encoding="utf-8-sig")
        if override_title:
            lines = text.splitlines()
            if lines and lines[0].startswith("#"):
                lines[0] = f"# {override_title}"
            else:
                lines.insert(0, f"# {override_title}")
            text = "\n".join(lines)
        buffer.append(text)

    if missing:
        print(f"WARN: missing chapters: {missing}", file=sys.stderr)

    combined = PAGE_BREAK.join(buffer) + "\n"
    OUT_MD.write_text(combined, encoding="utf-8")
    print(f"Wrote {OUT_MD}  ({len(combined.splitlines())} lines, {len(combined):,} chars)")

    script = ROOT / "scripts" / "md_to_thesis_docx.py"
    if not script.exists():
        print(f"WARN: {script} not found, skip docx", file=sys.stderr)
        return 0

    cmd = [sys.executable, str(script), "--input", str(OUT_MD), "--output", str(OUT_DOCX)]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return r.returncode
    print(f"Wrote {OUT_DOCX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
