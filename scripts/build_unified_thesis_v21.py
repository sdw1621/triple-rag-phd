"""
Build the unified PhD thesis v21 — 김원배 박사논문 (호서대, 同 advisor 문남미) 사례 추종.

vs v20 (작성지침 strict 해석):
  v20: 표지 → 사사 → 국문초록 → 본문 → 부록 → 참고문헌 → ABSTRACT
  v21: 표지 → 본문 → 부록 → 국문초록 → ABSTRACT → 참고문헌 → 감사의 글(사사)

또한:
  - 참고문헌 표기를 (N) → [N] 으로 복원 (김원배 사례 + IEEE 관례)
  - 본문 분류기호 변환은 v20 결과 그대로 재사용 (thesis_current/v5_rewrite/v20/)

근거:
  - 김원배 박사논문 (호서대 융합대학원, 문남미 교수 지도) 의 실제 통과본 분석
  - 작성지침 자) 의 "사사는 목차 앞" 은 권고이며, 실제 사례에선 맨 끝에 둠
  - 작성지침 (번호) 는 일반적 표기이며, [N] 도 광범위하게 인정됨

Usage:
    python scripts/build_unified_thesis_v21.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"
OUT_MD = V5 / "통합본_v21.md"
OUT_DOCX = V5 / "박사논문_통합본_v21.docx"

# Kim precedent order:
#   본문(Ⅰ~Ⅶ + CS) → 부록 → 국문초록 → ABSTRACT → 참고문헌 → 감사의 글
# 표지/속표지/인준서/연구윤리 서약서/목차는 한글 양식 .hwp 가 자체 처리.
ORDER = [
    ("00a_표지.md", None, V5),
    # ↓ 본문 시작 — 분류기호 변환된 v20/ 폴더 사용
    ("01_서론.md", None, V20),
    ("02_관련연구.md", None, V20),
    ("03_아키텍처.md", None, V20),
    ("04_R-DWA.md", None, V20),
    ("05_L-DWA.md", None, V20),
    ("06_실험평가.md", None, V20),
    ("06a_CS1-6_case_studies.md", None, V20),
    ("06b_CS7_list_prompt.md", None, V20),
    ("07_결론.md", None, V20),
    ("08_부록.md", None, V20),                        # 부록 (본문 직후)
    ("00b_국문초록.md", "국문초록", V5),               # 본문 끝 abstract block (KOR)
    ("00c_영문초록.md", "ABSTRACT", V5),               # 본문 끝 abstract block (ENG)
    # 09_참고문헌.md: V5 원본 사용 (대괄호 [N] 유지) — V20/ 의 (N) 변환본 사용 안 함
    ("09_참고문헌.md", "참고문헌", V5),
    ("10_감사의글.md", "감사의 글", V5),               # 사사 — 맨 끝 (김원배 사례)
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
