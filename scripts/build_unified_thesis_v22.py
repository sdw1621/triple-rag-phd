"""
Build the unified PhD thesis v22 — latest snapshot with all corrections applied.

vs v21 (Hoseo format with 김원배 precedent):
  본 v22 는 v21 과 동일한 구조를 유지하되, 다음 누적 정정이 모두 반영된
  최신 스냅샷이다:

  1. semantic Heading 1/2/3/4 styles (commit 0b216f7)
  2. stale sentence-prompt 수치 → corrected baseline (commit 9313990)
  3. paired bootstrap statistical defense + scope honesty (commit 3e8a4e2)
  4. Ch.1 서론 정합성 (commit ce41f4a)
       - Conditional 3-seed mean (0.285, +27.8%, Oracle 동등)
       - paired bootstrap 도입 명시 (§3 라)
       - List-prompt Uniform 향후 연구 항목 추가 (§4)
       - 표지 제출일 5월 정정
  5. References 100 entries fully translated to English (commit 40c8cfe)
  6. References 8 entries author corrections via arXiv 검증 (commit 64de61e)

구조 (v21 동일):
  표지 → 본문(Ⅰ~Ⅶ + CS) → 부록 → 국문초록 → ABSTRACT → 참고문헌 → 감사의 글
  표지/속표지/인준서/연구윤리 서약서/목차는 한글 양식 .hwp 가 자체 처리.

Usage:
    python scripts/build_unified_thesis_v22.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"
OUT_MD = V5 / "통합본_v22.md"
OUT_DOCX = V5 / "박사논문_통합본_v22.docx"

ORDER = [
    ("00a_표지.md", None, V5),
    # ↓ 본문 — 분류기호 변환된 v20/ 폴더 사용
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
    # 본문 직후 추록·참고문헌·사사
    ("00b_국문초록.md", "국문초록", V5),
    ("00c_영문초록.md", "ABSTRACT", V5),
    ("09_참고문헌.md", "References", V5),  # 영어 100항목 (commit 40c8cfe / 64de61e)
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
