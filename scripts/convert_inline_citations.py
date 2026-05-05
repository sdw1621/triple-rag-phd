"""
Convert body inline citations from "Author et al. (YEAR)" to IEEE "[N]" format.

The references file (09_참고문헌.md) uses IEEE numbered style [1]-[101]; the
body text was using author-year style ("Lewis et al. (2020)"), making the
mapping invisible to readers. This script replaces the author-year forms
with the matching [N] from the references list.

Examples:
    Lewis et al. (2020)            → [3]
    Asai et al. 2023               → [20]
    Schulman et al. (2017)         → [39]
    Pineau et al., 2021            → [97]
    Shin & Moon, 2025              → [1]
    Shin & Moon, 2025, JKSCI       → [1]   (venue annotation absorbed)

Korean particles attached after the citation are preserved naturally:
    "Lewis et al. (2020)의"        → "[3]의"

Skipped files (English content): 00c_영문초록.md, 09_참고문헌.md.

Usage:
    python scripts/convert_inline_citations.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5 = ROOT / "thesis_current" / "v5_rewrite"
V20 = V5 / "v20"

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
    V5 / "00b_국문초록.md",
    V5 / "10_감사의글.md",
]

# Mapping: (regex, replacement, label)
# Order matters — longest/most-specific patterns FIRST so they shadow
# shorter alternatives.
PATTERNS: list[tuple[str, str, str]] = [
    # ----- Multi-author with venue annotation -----
    (r"Shin\s*&\s*Moon,?\s*\(?2025\)?,?\s*JKSCI", "[1]", "Shin & Moon, 2025, JKSCI"),
    (r"Shin\s+and\s+Moon,?\s*\(?2025\)?,?\s*JKSCI", "[1]", "Shin and Moon, 2025, JKSCI"),
    # ----- Multi-author standard forms -----
    (r"Shin\s*&\s*Moon,?\s*\(?2025\)?", "[1]", "Shin & Moon, 2025"),
    (r"Shin\s+and\s+Moon,?\s*\(?2025\)?", "[1]", "Shin and Moon, 2025"),
    # ----- Single-author "et al." patterns: (Author et al.,? (YEAR)) -----
    # Each pattern: optional comma between "al." and year, optional parens around year.
    (r"Lewis\s+et\s+al\.,?\s*\(?2020\)?", "[3]", "Lewis et al. 2020"),
    (r"Karpukhin\s+et\s+al\.,?\s*\(?2020\)?", "[6]", "Karpukhin et al. 2020"),
    (r"Asai\s+et\s+al\.,?\s*\(?202[34]\)?", "[20]", "Asai et al. 2023/2024"),
    (r"Yan\s+et\s+al\.,?\s*\(?2024\)?", "[21]", "Yan et al. 2024"),
    (r"Madaan\s+et\s+al\.,?\s*\(?2023\)?", "[22]", "Madaan et al. 2023"),
    (r"Shinn\s+et\s+al\.,?\s*\(?2023\)?", "[23]", "Shinn et al. 2023"),
    (r"Yao\s+et\s+al\.,?\s*\(?2023\)?", "[25]", "Yao et al. 2023"),
    (r"Jeong\s+et\s+al\.,?\s*\(?2024\)?", "[26]", "Jeong et al. 2024"),
    (r"Press\s+et\s+al\.,?\s*\(?2023\)?", "[27]", "Press et al. 2023"),
    (r"Mallen\s+et\s+al\.,?\s*\(?2023\)?", "[28]", "Mallen et al. 2023"),
    (r"Trivedi\s+et\s+al\.,?\s*\(?2023\)?", "[30]", "Trivedi et al. 2023 (IRCoT)"),
    (r"Edge\s+et\s+al\.,?\s*\(?2024\)?", "[13]", "Edge et al. 2024"),
    (r"Sun\s+et\s+al\.,?\s*\(?2024\)?", "[15]", "Sun et al. 2024"),
    (r"Khattab\s+(?:and|&)\s+Zaharia,?\s*\(?2020\)?", "[32]", "Khattab & Zaharia 2020"),
    (r"Schulman\s+et\s+al\.,?\s*\(?2017\)?", "[39]", "Schulman et al. 2017"),
    (r"Schulman\s+et\s+al\.,?\s*\(?2016\)?", "[38]", "Schulman et al. 2016"),
    (r"Schulman\s+et\s+al\.,?\s*\(?2015\)?", "[37]", "Schulman et al. 2015"),
    (r"Mnih\s+et\s+al\.,?\s*\(?2013\)?", "[40]", "Mnih et al. 2013"),
    (r"Mnih\s+et\s+al\.,?\s*\(?2016\)?", "[41]", "Mnih et al. 2016"),
    (r"Christiano\s+et\s+al\.,?\s*\(?2017\)?", "[47]", "Christiano et al. 2017"),
    (r"Ouyang\s+et\s+al\.,?\s*\(?2022\)?", "[49]", "Ouyang et al. 2022"),
    (r"Rafailov\s+et\s+al\.,?\s*\(?2023\)?", "[50]", "Rafailov et al. 2023"),
    (r"Yuan\s+et\s+al\.,?\s*\(?2024\)?", "[52]", "Yuan et al. 2024"),
    (r"Brown\s+et\s+al\.,?\s*\(?2020\)?", "[54]", "Brown et al. 2020"),
    (r"Touvron\s+et\s+al\.,?\s*\(?2023\)?", "[56]", "Touvron et al. 2023"),
    (r"Devlin\s+et\s+al\.,?\s*\(?2019\)?", "[59]", "Devlin et al. 2019"),
    (r"Liu\s+et\s+al\.,?\s*\(?2019\)?", "[60]", "Liu et al. 2019"),
    (r"Vaswani\s+et\s+al\.,?\s*\(?2017\)?", "[61]", "Vaswani et al. 2017"),
    (r"Reimers\s+(?:and|&)\s+Gurevych,?\s*\(?2019\)?", "[62]", "Reimers & Gurevych 2019"),
    (r"Park\s+et\s+al\.,?\s*\(?2021\)?", "[65]", "Park et al. 2021 (KLUE)"),
    (r"Yang\s+et\s+al\.,?\s*\(?2018\)?", "[70]", "Yang et al. 2018 (HotpotQA)"),
    (r"Trivedi\s+et\s+al\.,?\s*\(?2022\)?", "[71]", "Trivedi et al. 2022 (MuSiQue)"),
    (r"Jin\s+et\s+al\.,?\s*\(?2019\)?", "[72]", "Jin et al. 2019 (PubMedQA)"),
    (r"Rajpurkar\s+et\s+al\.,?\s*\(?2016\)?", "[76]", "Rajpurkar et al. 2016 (SQuAD)"),
    (r"Maynez\s+et\s+al\.,?\s*\(?2020\)?", "[84]", "Maynez et al. 2020"),
    (r"Ji\s+et\s+al\.,?\s*\(?2023\)?", "[85]", "Ji et al. 2023"),
    (r"Wei\s+et\s+al\.,?\s*\(?2022\)?", "[89]", "Wei et al. 2022 (CoT)"),
    (r"Kojima\s+et\s+al\.,?\s*\(?2022\)?", "[90]", "Kojima et al. 2022"),
    (r"Schick\s+et\s+al\.,?\s*\(?2023\)?", "[93]", "Schick et al. 2023"),
    (r"Pineau\s+et\s+al\.,?\s*\(?2021\)?", "[97]", "Pineau et al. 2021"),
    (r"Henderson\s+et\s+al\.,?\s*\(?2018\)?", "[98]", "Henderson et al. 2018"),
    (r"Colas\s+et\s+al\.,?\s*\(?2018\)?", "[101]", "Colas et al. 2018"),
]


def convert(text: str) -> tuple[str, dict[str, int]]:
    stats: dict[str, int] = {}
    for pat, repl, label in PATTERNS:
        text, n = re.subn(pat, repl, text)
        if n:
            stats[label] = n
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
            count = sum(stats.values())
            print(f"  ✓ {path.relative_to(ROOT)}: {count} replacements {stats}")
            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v

    print()
    print(f"Files changed: {files_changed}")
    print(f"Total inline-citation replacements:")
    for k, v in sorted(total_stats.items(), key=lambda x: -x[1]):
        print(f"  {v:3d}  {k}")
    print(f"  ----")
    print(f"  {sum(total_stats.values()):3d}  TOTAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
