"""
Convert inline 'Ch.X' / '§Y' references to thesis-style Roman + 절.

Examples:
    Ch.5 §3        → Ⅴ장 3절
    Ch.6 §3.4      → Ⅵ장 3.4절
    Ch.6 §3 아     → Ⅵ장 3절 아   (preserve subsection letter 가/나/다/...)
    Ch.5           → Ⅴ장          (standalone)
    §3             → 3절           (standalone)
    §3.6           → 3.6절
    §3 아          → 3절 아

Skipped files (English content): 00c_영문초록.md, 09_참고문헌.md.

Usage:
    python scripts/convert_chapter_section.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"

# Files in Korean — apply conversion
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
    V5 / "00a_표지.md",
    V5 / "00b_국문초록.md",
    V5 / "10_감사의글.md",
]

ROMAN = {1: "Ⅰ", 2: "Ⅱ", 3: "Ⅲ", 4: "Ⅳ", 5: "Ⅴ", 6: "Ⅵ", 7: "Ⅶ"}
SUB_LETTER_CLASS = "가-하"  # Korean subsection letters 가/나/다/라/마/바/사/아/자/차/카/타/파/하


def convert(text: str) -> tuple[str, dict[str, int]]:
    """Apply conversions in priority order. Returns (new_text, stats)."""
    stats: dict[str, int] = {}

    def _track(label: str, count: int) -> None:
        if count:
            stats[label] = stats.get(label, 0) + count

    def _ch(num_str: str) -> str:
        return ROMAN[int(num_str)] + "장"

    # 1. Ch.X §Y.Z [가/나/...]   →   Ⅹ장 Y.Z절 W
    pat1 = re.compile(
        rf"Ch\.([1-7]) §(\d+\.\d+) ([{SUB_LETTER_CLASS}])(?=[\s.,)])"
    )
    text, n = pat1.subn(lambda m: f"{_ch(m.group(1))} {m.group(2)}절 {m.group(3)}", text)
    _track("Ch.X §Y.Z W", n)

    # 2. Ch.X §Y [가/나/...]   →   Ⅹ장 Y절 W
    pat2 = re.compile(
        rf"Ch\.([1-7]) §(\d+) ([{SUB_LETTER_CLASS}])(?=[\s.,)])"
    )
    text, n = pat2.subn(lambda m: f"{_ch(m.group(1))} {m.group(2)}절 {m.group(3)}", text)
    _track("Ch.X §Y W", n)

    # 3. Ch.X §Y.Z   →   Ⅹ장 Y.Z절
    pat3 = re.compile(r"Ch\.([1-7]) §(\d+\.\d+)")
    text, n = pat3.subn(lambda m: f"{_ch(m.group(1))} {m.group(2)}절", text)
    _track("Ch.X §Y.Z", n)

    # 4. Ch.X §Y   →   Ⅹ장 Y절
    pat4 = re.compile(r"Ch\.([1-7]) §(\d+)")
    text, n = pat4.subn(lambda m: f"{_ch(m.group(1))} {m.group(2)}절", text)
    _track("Ch.X §Y", n)

    # 5. §Y.Z [가/나/...]   →   Y.Z절 W   (standalone)
    pat5 = re.compile(rf"§(\d+\.\d+) ([{SUB_LETTER_CLASS}])(?=[\s.,)])")
    text, n = pat5.subn(lambda m: f"{m.group(1)}절 {m.group(2)}", text)
    _track("§Y.Z W", n)

    # 6. §Y [가/나/...]   →   Y절 W   (standalone)
    pat6 = re.compile(rf"§(\d+) ([{SUB_LETTER_CLASS}])(?=[\s.,)])")
    text, n = pat6.subn(lambda m: f"{m.group(1)}절 {m.group(2)}", text)
    _track("§Y W", n)

    # 7. §Y.Z   →   Y.Z절   (standalone)
    pat7 = re.compile(r"§(\d+\.\d+)")
    text, n = pat7.subn(lambda m: f"{m.group(1)}절", text)
    _track("§Y.Z", n)

    # 8. §Y   →   Y절   (standalone)
    pat8 = re.compile(r"§(\d+)")
    text, n = pat8.subn(lambda m: f"{m.group(1)}절", text)
    _track("§Y", n)

    # 9. Ch.X (standalone, no §) — match even when followed by Korean particle
    # like 의/은/는/이/가/에/에서 etc. Just require word-boundary BEFORE Ch.
    pat9 = re.compile(r"\bCh\.([1-7])")
    text, n = pat9.subn(lambda m: _ch(m.group(1)), text)
    _track("Ch.X (standalone)", n)

    # 10-13. §<Roman>.<section[.subsection]> — pattern from older drafts
    # §Ⅴ.2  →  Ⅴ장 2절
    # §Ⅵ.1.4 →  Ⅵ장 1.4절
    roman_to_arabic = {"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5", "Ⅵ": "6", "Ⅶ": "7"}
    roman_class = "ⅠⅡⅢⅣⅤⅥⅦ"

    # 10. §Ⅹ.Y.Z → Ⅹ장 Y.Z절
    pat10 = re.compile(rf"§([{roman_class}])\.(\d+\.\d+)")
    text, n = pat10.subn(lambda m: f"{m.group(1)}장 {m.group(2)}절", text)
    _track("§Ⅹ.Y.Z", n)

    # 11. §Ⅹ.Y → Ⅹ장 Y절
    pat11 = re.compile(rf"§([{roman_class}])\.(\d+)")
    text, n = pat11.subn(lambda m: f"{m.group(1)}장 {m.group(2)}절", text)
    _track("§Ⅹ.Y", n)

    # 12. "본 §은", "본 §의", "이 §" 같은 standalone § (no number) — drop §
    # by replacing with appropriate Korean noun (절 in body context)
    pat12 = re.compile(r"본 §([은는이가의를])")
    text, n = pat12.subn(lambda m: f"본 절{m.group(1)}", text)
    _track("본 §은/의/...", n)

    # 13. Ch.Ⅹ (Roman after Ch.) — leftover from earlier mixed-style drafts
    pat13 = re.compile(rf"Ch\.([{roman_class}])")
    text, n = pat13.subn(lambda m: f"{m.group(1)}장", text)
    _track("Ch.Ⅹ", n)

    return text, stats


def main() -> int:
    total_stats: dict[str, int] = {}
    files_changed = 0

    for path in KOR_FILES:
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        original = path.read_text(encoding="utf-8")
        new_text, stats = convert(original)
        if new_text != original:
            path.write_text(new_text, encoding="utf-8")
            files_changed += 1
            total_stats_count = sum(stats.values())
            print(f"  ✓ {path.relative_to(ROOT)}: {total_stats_count} replacements {stats}")
            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v

    print()
    print(f"Files changed: {files_changed}")
    print(f"Total replacements:")
    for k, v in sorted(total_stats.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")
    print(f"  ----")
    print(f"  {sum(total_stats.values()):4d}  TOTAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
