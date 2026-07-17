# 📄 Thesis Drafts — ⚠️ SUPERSEDED (historical reference only)

> ## ⚠️ 이 디렉터리는 최종본이 아닙니다 (2026-07-17 기준)
>
> **논문 최종본은 저장소 밖의 한글 문서입니다: `260716_박사학위논문_신동욱.hwpx`**
>
> 이 디렉터리의 `.docx` / `.md` 원고와 `scripts/build_unified_thesis_*.py` 빌드
> 파이프라인은 **hwpx 직접 편집으로 전환하기 이전의 이력 자료**입니다.
> 이후의 심사 반영 사항(관련연구 확장, 참고문헌 재구성, 그림 교체 등)은
> hwpx 에만 반영되어 있으므로, 이 마크다운에서 논문을 재생성하면
> **최신 작업이 모두 사라집니다.**
>
> ### 용어 주의 — L-DWA → A-DWA
> 최종본은 알고리즘 명칭을 **A-DWA**(Adaptive Dynamic Weighting Algorithm)로
> 확정했습니다. 이 디렉터리의 원고는 이전 명칭인 **L-DWA** 를 그대로 쓰고 있으며,
> 이력 보존을 위해 일괄 치환하지 않았습니다.
> 파일명(`05_L-DWA.md`, `박사논문_5장_PPO_LDWA_v4.docx`)도 같은 이유로 유지합니다.
>
> `scripts/plot_figure*.py` 와 `docs/figures/` 는 A-DWA 로 통일되어 있습니다.

---

## 아래 내용은 전환 이전(2026-04-19) 기준으로 작성된 것입니다

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
