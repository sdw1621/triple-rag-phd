"""
Renumber PhD thesis markdown headings to Hoseo PhD format.

Hoseo 양식 분류기호 (작성지침 제9조):
  장(章): Ⅰ. Ⅱ. ... (handled at H1)
  절(節): 1. 2. ... (H2)
  항(項): 가. 나. ... (H3, sequential within parent H2)
  목(目): a. b. ... (H4, sequential within parent H3)

Transformations:
  # 제Ⅰ장 서론 (v5 재작성)        → # Ⅰ. 서론
  ## 1절 연구 배경                  → ## 1. 연구 배경
  ### 1.1 하드웨어                  → ### 가. 하드웨어
  ### 3.1 ...; ### 3.2 ... (dup)   → ### 가. ...; ### 나. ... (renumber sequentially)
  #### HotpotQA                    → #### a. HotpotQA

Special:
  - ## 변경 로그 (v4 → v5) and its body are stripped (dev meta, not for submission)
  - ## without "N절" pattern (e.g., "본 장 핵심 기여의 위치", "## CS-1.") preserved as-is
  - Appendix file (08_부록.md) uses --keep-appendix to skip renumbering
  - Code blocks (``` … ```) are not touched

Usage:
  python scripts/renumber_to_hoseo.py --in 01_서론.md --out 01_서론_v20.md
  python scripts/renumber_to_hoseo.py --in 08_부록.md --out 08_부록_v20.md --keep-appendix
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KO_LETTERS = list("가나다라마바사아자차카타파하")


def ko(n: int) -> str:
    return KO_LETTERS[n - 1] if 1 <= n <= len(KO_LETTERS) else f"({n})"


def al(n: int) -> str:
    return chr(ord("a") + n - 1) if 1 <= n <= 26 else f"({n})"


H_RE = re.compile(r"^(#+)\s+(.*?)\s*$")
SECTION_RE = re.compile(r"^(\d+)\s*절\s+(.*)$")
SUBSECTION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?\s+(.*)$")
CHAPTER_RE = re.compile(r"^제\s*([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\s*장\s+(.*)$")
V5_TAG_RE = re.compile(r"\s*\((?:v[345](?:\s*재작성)?|v\d+\s*→\s*v\d+)\)\s*$")


def strip_dev_tags(title: str) -> str:
    return V5_TAG_RE.sub("", title).strip()


def renumber(text: str) -> str:
    """Apply Hoseo numbering to chapter markdown.

    Counters (h3_count, h4_count) are reset whenever an H2 is encountered (numbered or not).
    """
    out: list[str] = []
    h3 = 0  # current count within parent H2
    h4 = 0  # current count within parent H3
    in_code = False
    skip_block = False  # for ## 변경 로그 — drop until next H2 / EOF

    for line in text.splitlines():
        # code fences
        if line.startswith("```"):
            in_code = not in_code
            if not skip_block:
                out.append(line)
            continue
        if in_code:
            if not skip_block:
                out.append(line)
            continue

        m = H_RE.match(line)
        if not m:
            if not skip_block:
                out.append(line)
            continue

        level = len(m.group(1))
        title = m.group(2)

        if level == 1:
            skip_block = False
            title = strip_dev_tags(title)
            cm = CHAPTER_RE.match(title)
            if cm:
                title = f"{cm.group(1)}. {cm.group(2)}"
            out.append(f"# {title}")
            h3 = 0
            h4 = 0
            continue

        if level == 2:
            # decide whether this H2 is the dev-meta block
            if title.startswith("변경 로그"):
                skip_block = True
                continue
            skip_block = False
            sm = SECTION_RE.match(title)
            if sm:
                out.append(f"## {sm.group(1)}. {sm.group(2)}")
            else:
                # Non-numbered H2 (e.g., "본 장 핵심 기여의 위치", "CS-1. ...")
                out.append(f"## {title}")
            h3 = 0
            h4 = 0
            continue

        if level == 3:
            if skip_block:
                continue
            sm = SUBSECTION_RE.match(title)
            if sm:
                rest = sm.group(1)
            else:
                rest = title
            h3 += 1
            h4 = 0
            out.append(f"### {ko(h3)}. {rest}")
            continue

        if level >= 4:
            if skip_block:
                continue
            h4 += 1
            # strip leading "N.M.K " if present
            rest = re.sub(r"^\d+\.\d+\.\d+\s+", "", title)
            out.append(f"#### {al(h4)}. {rest}")
            continue

        # fallback (should not reach)
        if not skip_block:
            out.append(line)

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def light_pass(text: str) -> str:
    """For appendix files: only strip dev tags, keep numbering."""
    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        m = H_RE.match(line)
        if m and len(m.group(1)) == 1:
            out.append(f"# {strip_dev_tags(m.group(2))}")
        else:
            out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="input", required=True, type=Path)
    ap.add_argument("--out", dest="output", required=True, type=Path)
    ap.add_argument("--keep-appendix", action="store_true",
                    help="Skip renumbering (only strip dev tags). For 08_부록.md.")
    args = ap.parse_args()

    src = args.input.read_text(encoding="utf-8-sig")  # auto-strip BOM
    dst = light_pass(src) if args.keep_appendix else renumber(src)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dst, encoding="utf-8")
    print(f"Wrote {args.output} ({len(dst.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
