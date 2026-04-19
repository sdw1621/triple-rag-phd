# 📄 Current Thesis Drafts (v4/v3)

> These are the current versions of the PhD thesis chapters as of 2026-04-19.
> Claude Code should reference these for thesis context but NOT modify them.
> Modifications are done on Claude Desktop (web chat) by the author.

---

## 📂 Files

| File | Chapter | Version | Status |
|---|---|---|---|
| `박사논문_1장_서론_v4.docx` | Ⅰ. 서론 | v4 | ✅ Final |
| `박사논문_2장_관련연구_v4.docx` | Ⅱ. 관련 연구 | v4 | ✅ Final |
| `박사논문_3장_TripleHybridArchitecture_v4.docx` | Ⅲ. Triple-Hybrid 아키텍처 | v4 | ✅ Final |
| `박사논문_4장_RuleBasedDWA_v4.docx` | Ⅳ. R-DWA | v4 | ✅ Final |
| `박사논문_5장_PPO_LDWA_v4.docx` | Ⅴ. PPO 기반 L-DWA | v4 | ✅ Final |
| `박사논문_6장_실험평가_v2.docx` | Ⅵ. 실험 및 평가 | v2 | ⏳ § placeholders to replace |
| `박사논문_6장_실험평가_v3_확장섹션.docx` | Ⅵ. 확장 섹션 | v3 | ⏳ § placeholders to replace |
| `박사논문_7장_결론_v3.docx` | Ⅶ. 결론 | v3 | ✅ Final |

---

## 🎯 Purpose

Claude Code can read these to:
- Understand exact thesis claims when implementing code
- Verify code specifications match thesis descriptions
- Cross-reference equation numbers (e.g., Eq. 5-7)
- Generate implementation that matches thesis tables

---

## ❌ DO NOT Modify

- These are author-curated by Claude Desktop
- Any content changes happen there, not in Claude Code
- Code implementations reference these, not replace them

## ✅ DO Reference

When implementing a module, consult the relevant chapter:
- Port prior code → Ch.3, Ch.4 (Triple-Hybrid, R-DWA)
- BERT Intent → Ch.3 Sec 3
- PPO + L-DWA → Ch.5 (critical!)
- Evaluation → Ch.6
- Future work → Ch.7

---

## 🔄 How to Read docx in Container

```bash
# Inside container
docker-compose exec triple_rag python -c "
from docx import Document
doc = Document('/workspace/thesis_current/박사논문_5장_PPO_LDWA_v4.docx')
for p in doc.paragraphs[:50]:
    if p.text.strip():
        print(p.text)
"
```

Or use `scripts/read_thesis.py` (create as needed).
