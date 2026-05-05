"""
Build the unified PhD thesis v23 — latest snapshot with thesis-style numbering.

vs v22:
  v22: inline references use "Ch.5 §3" (English shorthand + § silcrow)
  v23: inline references use "Ⅴ장 3절" (Korean thesis convention,
       Roman numerals + 절)

  변환된 패턴 (총 ~130 위치):
    Ch.5 §3       → Ⅴ장 3절
    Ch.6 §3.4     → Ⅵ장 3.4절
    Ch.6 §3 아    → Ⅵ장 3절 아   (subsection letter 보존)
    Ch.5          → Ⅴ장          (standalone)
    §3            → 3절          (standalone)
    §Ⅴ.2         → Ⅴ장 2절      (legacy pattern)

  변환 적용 파일 (Korean):
    v20/01_서론.md ~ 08_부록.md (8 files)
    00a_표지.md, 00b_국문초록.md, 10_감사의글.md
  미적용 (English):
    00c_영문초록.md, 09_참고문헌.md (영어 100항목)

vs v22 누적 정정 (모두 v23 에 포함):
  1. semantic Heading 1/2/3/4 styles (commit 0b216f7)
  2. stale sentence-prompt 수치 → corrected baseline (9313990)
  3. paired bootstrap statistical defense + scope honesty (3e8a4e2)
  4. Ch.1 서론 정합성 (ce41f4a)
  5. References 100 entries fully translated to English (40c8cfe)
  6. References 8 entries author corrections via arXiv (64de61e)
  7. blockquote 내부 markdown 표/HTML 태그 fix (f544698)

구조 (v22/v21 동일):
  표지 → 본문(Ⅰ~Ⅶ + CS) → 부록 → 국문초록 → ABSTRACT → 참고문헌 → 감사의 글

Usage:
    python scripts/build_unified_thesis_v23.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"
OUT_MD = V5 / "통합본_v23.md"
OUT_DOCX = V5 / "박사논문_통합본_v23.docx"

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
