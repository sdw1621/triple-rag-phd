"""
Build the unified PhD thesis v19 markdown + docx.

v19 vs v18 (humanize round):
  - Ch.6 §1.3 / §1.4 / §2.1-§2.3 : bold-heavy AI tone → prose flow
  - Ch.7 §3-§4 (학술적·실용적 의의, 한계, 향후 연구) : 번호 목록/굵은 소제목 축소,
    자연스러운 문단 서술로 재작성
  - Ch.5 §4 (오프라인 보상 캐시) 는 v18 단계에서 이미 humanize 반영됨
  - CS-1~6, CS-7 케이스 스터디는 v18 단계에서 완료된 상태 유지

Usage:
    python scripts/build_unified_thesis_v19.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
OUT_MD = V5 / "통합본_v19.md"
OUT_DOCX = V5 / "박사논문_통합본_v19.docx"

# Order matches Hoseo 박사논문 규격
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
    ("09_참고문헌.md", None),
    ("10_감사의글.md", None),
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

    cmd = [
        sys.executable,
        str(script),
        "--input", str(OUT_MD),
        "--output", str(OUT_DOCX),
        "--chapter", "박사학위 논문 통합본 v19",
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
