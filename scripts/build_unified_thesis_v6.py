"""
Build the unified PhD thesis v6 markdown + docx.

Reads the individual chapter markdowns in thesis_current/v5_rewrite/,
concatenates them in the correct order (front matter → 본문 1~7 →
case studies → 부록), then runs md_to_thesis_docx.py on the combined
file to produce a single .docx matching Hoseo format.

v6 vs v5:
  - Ch.3 / Ch.5 / Ch.6 now include the 9 definition+example boxes
  - Ch.6 §3.3 extended with list-prompt reproduction row
  - Ch.6 §3.4 / §3.5 (list-prompt analysis + faithfulness guide) new
  - 06a (CS-1~6) and 06b (CS-7) case studies new
  - 부록 A~F new

Usage:
    python scripts/build_unified_thesis_v6.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
OUT_MD = V5 / "통합본_v6.md"
OUT_DOCX = V5 / "박사논문_통합본_v6.docx"

# Order matters — must match Hoseo 박사논문 구성
ORDER = [
    ("00a_표지.md", None),
    ("00b_국문초록.md", "국문초록"),
    ("00c_영문초록.md", "ABSTRACT"),
    ("00d_목차.md", None),
    ("01_서론.md", None),
    ("02_관련연구.md", None),
    ("03_아키텍처.md", None),
    ("04_R-DWA.md", None),
    ("05_L-DWA.md", None),
    ("06_실험평가.md", None),
    ("06a_CS1-6_case_studies.md", None),
    ("06b_CS7_list_prompt.md", None),
    ("07_결론.md", None),
    ("08_부록.md", None),
]

PAGE_BREAK = "\n\n---\n\n"


def main() -> int:
    buffer: list[str] = []
    missing: list[str] = []

    for fname, override_title in ORDER:
        p = V5 / fname
        if not p.exists():
            missing.append(fname)
            continue
        text = p.read_text(encoding="utf-8")
        if override_title:
            # replace leading heading if present
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

    # docx conversion
    script = ROOT / "scripts" / "md_to_thesis_docx.py"
    if not script.exists():
        print(f"WARN: {script} not found, skip docx", file=sys.stderr)
        return 0

    cmd = [
        sys.executable,
        str(script),
        "--input", str(OUT_MD),
        "--output", str(OUT_DOCX),
        "--chapter", "박사학위 논문 통합본 v6",
    ]
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
