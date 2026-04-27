"""
Build the unified PhD thesis v20 (Hoseo 박사논문 양식 호환).

vs v19:
  - Heading numbering: Ⅰ./1./가./a. (작성지침 제9조)
  - Front-matter reordered:
      v19:  표지 → 국문초록 → ABSTRACT → 목차 → 본문 → … → 참고문헌 → 감사의 글
      v20:  표지 → 감사의 글 → 국문초록 → (목차 placeholder) → 본문 → 부록 → 참고문헌 → ABSTRACT
  - References: [N] → (N), category H2 brackets dropped
  - 학번 placeholder filled (20215175)

The output 통합본_v20.docx is intended to be imported into the Hoseo .hwp
양식. The Hangul 양식 template handles 표지/속표지/인준서/연구윤리 서약서/목차.
The user pastes the body content (사사 ~ ABSTRACT) into the template's body area.

Usage:
    python scripts/build_unified_thesis_v20.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"
OUT_MD = V5 / "통합본_v20.md"
OUT_DOCX = V5 / "박사논문_통합본_v20.docx"

# Hoseo 박사논문 양식 — 작성지침 제9조 + 국문샘플 준수
# 표지/속표지/인준서/연구윤리 서약서/목차는 한글 .hwp 양식 페이지가 처리
ORDER = [
    ("00a_표지.md", None, V5),                           # 표지 (서식1) — 양식엔 학번 안 들어감
    ("10_감사의글.md", "감사의 글", V5),                # 사사 (작성지침 자, 목차 앞)
    ("00b_국문초록.md", "국문초록", V5),                # 국문초록
    # ↓ 목차 자리: 한글 양식이 자동 생성 (heading 스타일에서 추출)
    ("01_서론.md", None, V20),
    ("02_관련연구.md", None, V20),
    ("03_아키텍처.md", None, V20),
    ("04_R-DWA.md", None, V20),
    ("05_L-DWA.md", None, V20),
    ("06_실험평가.md", None, V20),
    ("06a_CS1-6_case_studies.md", None, V20),
    ("06b_CS7_list_prompt.md", None, V20),
    ("07_결론.md", None, V20),
    ("08_부록.md", None, V20),                          # 부록 (작성지침 사)
    ("09_참고문헌.md", "참고문헌", V20),                # 참고문헌 (작성지침 아)
    ("00c_영문초록.md", "ABSTRACT", V5),                # 영문초록 (작성지침 마, 논문 뒤편)
]

PAGE_BREAK = "\n\n---\n\n"


REF_BRACKET_RE = re.compile(r"^\[(\d+)\]\s")
REF_CATEGORY_RE = re.compile(r"^##\s+\[\d+\]\s+(.*?)\s*$")


def convert_references(text: str) -> str:
    """09_참고문헌.md: [N] → (N), drop [N] from category H2 headers."""
    out: list[str] = []
    for line in text.splitlines():
        cm = REF_CATEGORY_RE.match(line)
        if cm:
            out.append(f"## {cm.group(1)}")
            continue
        rm = REF_BRACKET_RE.match(line)
        if rm:
            out.append(f"({rm.group(1)}) " + line[rm.end():])
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main() -> int:
    V20.mkdir(parents=True, exist_ok=True)

    # Pre-write references file into v20/ with bracket conversion
    refs_src = V5 / "09_참고문헌.md"
    refs_dst = V20 / "09_참고문헌.md"
    if refs_src.exists():
        refs_dst.write_text(
            convert_references(refs_src.read_text(encoding="utf-8-sig")),
            encoding="utf-8",
        )
        print(f"Wrote {refs_dst}")

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

    cmd = [
        sys.executable,
        str(script),
        "--input", str(OUT_MD),
        "--output", str(OUT_DOCX),
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
