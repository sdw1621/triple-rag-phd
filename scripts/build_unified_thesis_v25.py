"""
Build the unified PhD thesis v25 — self-reference removed, references renumbered.

vs v24:
  v24: References had [2] D.-W. Shin (this dissertation) — non-standard
       self-reference; entries [1]–[101].
  v25: [2] removed, [3]–[101] renumbered to [2]–[100]. Body inline citations
       shifted accordingly (17 inline citations updated).

  - References: 101 → 100 entries.
  - Body: [N] (N>=3) shifted to [N-1] in 4 files (02, 05, 08, 00b).

vs v24/v23/v22 누적 정정 (모두 v25 에 포함):
  1. semantic Heading 1/2/3/4 styles
  2. stale → corrected baseline
  3. paired bootstrap statistical defense
  4. Ch.1 서론 정합성 (3-seed mean, paired bootstrap 언급)
  5. References 100 entries 영어 + 8 author corrections
  6. blockquote 내부 markdown 표 fix
  7. Ⅰ장 N절 표기 통일 (132건)
  8. IEEE [N] inline citations + Model[N] 형식 통일
  9. [2] self-reference 제거 + renumber

Usage:
    python scripts/build_unified_thesis_v25.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"
OUT_MD = V5 / "통합본_v25.md"
OUT_DOCX = V5 / "박사논문_통합본_v25.docx"

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
