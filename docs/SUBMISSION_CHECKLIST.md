# 🎓 박사학위 논문 제출 체크리스트 (2026-04-30 D-8)

**작성 2026-04-22 22:00 KST**
**목적**: 제출 시점(4/30) 전까지 남은 작업과 완료 작업을 한눈에. 외출 / 수면에서 돌아왔을 때 여기서부터 시작.

---

## ✅ 완료 (PR 단위)

| PR | 머지 | 내용 |
|---|---|---|
| #5 | ✅ | list-prompt reproducibility defense + 9 definition boxes + CS-7 |
| #6 | ✅ | Streamlit 대시보드 (4 탭, zero LLM cost) |
| #7 | ✅ | 통합본 v6 + Fig 6-4 (λ sensitivity) + Fig 6-5 (cross-domain radar) |
| #8 | ✅ | Ch.1/2/4/7 박스 (정의+예시 체계 균일화) |
| #9 | ❌ closed | → #10 에 포함 |
| #10 | ✅ | Cross-lingual A+B (영어 intent + 영어 합성 벤치) |
| #11 | ✅ | 초록 재작성 + 통합본 v10 final |

## 📊 핵심 수치 (Final, 모든 문서에서 일관)

| 지표 | R-DWA (corrected) | **L-DWA (3-seed)** | Oracle | 비고 |
|---|---|---|---|---|
| F1<sub>strict</sub> | 0.529 | **0.562 ± 0.007** | 0.554 | **Oracle 초과** |
| F1<sub>substring</sub> | 0.482 | **0.507 ± 0.009** | 0.504 | **Oracle 초과** |
| F1<sub>char</sub> | 0.469 | **0.494 ± 0.010** | 0.487 | **Oracle 초과** |
| EM | 0.387 | 0.388 ± 0.001 | 0.388 | ≈ Oracle |
| Faithfulness | 0.544 | **0.580 ± 0.009** | 0.570 | **Oracle 초과** |

**서사**: 선행 JKSCI 재현 장애 3 정정 → corrected baseline 수립 → L-DWA 가 Oracle 초과.

## 🗂 산출물 현황

### 코드
- `src/` — 모든 핵심 모듈 (평가기, 프롬프트, intent 한/영 듀얼, L-DWA, PPO)
- `scripts/` — build_cache, train_ppo, evaluate_rerun, aggregate, 5개 plot 스크립트
- `streamlit_app/dashboard.py` — 4-tab 대시보드 (포트 8501)
- `tests/test_eval_metrics.py` — 37 tests pass
- `data/university_en/` — 영어 합성 벤치 (403 QA, 414 docs)

### 결과
- `results/rerun_*_list.json` — 5,000 QA × 5 정책 list-prompt rerun
- `results/english_synthetic_rdwa.json` — 영어 합성 R-DWA 평가
- `results/cross_hotpotqa_*_EN_intent.json` — 영어 intent HotpotQA 재측정
- `results/m8_list_prompt_summary.md` — 자동 생성 요약

### 피규어 (6개)
- `docs/figures/fig3_1_pipeline.png` — Triple-Hybrid 파이프라인
- `docs/figures/fig5_1_actor_critic.png` — Actor-Critic 구조
- `docs/figures/fig5_2_ppo_convergence.png` — 3-seed 수렴 곡선
- `docs/figures/fig6_1_prompt_delta.png` — sentence vs list 대비
- `docs/figures/fig6_2_per_type.png` — per-type F1
- `docs/figures/fig6_3_weight_distribution.png` — Δ³ 가중치 분포
- `docs/figures/fig6_4_lambda_sensitivity.png` — λ 민감도
- `docs/figures/fig6_5_cross_domain_radar.png` — 교차 도메인

### 논문 (호서대 규격 docx)
- `thesis_current/v5_rewrite/박사논문_통합본_v10_final.docx` — **2,117줄 / 103 KB, 제출 후보**
- 각 장 개별 docx: 01_서론, 02_관련연구, 03_아키텍처, 04_R-DWA, 05_L-DWA, 06_실험평가, 06a_CS1-6, 06b_CS7, 07_결론, 08_부록
- 박스 13개 (Ch.1~7 전부 포괄)
- 케이스 스터디 13개 (CS-1~CS-6 + CS-7)
- 부록 A~F (하이퍼파라미터, 의사코드, 버그 기록, 실패 실험, 자원, 재현)

### 문서
- `docs/CORRECTED_BASELINE.md` — stage-wise 기준점 단일 출처
- `docs/M8_REPRODUCIBILITY_DEFENSE.md` — 방어 전략 문서
- `docs/SUBMISSION_CHECKLIST.md` — **이 문서**
- 선행 저장소 `sdw1621/hybrid-rag-comparsion` CORRIGENDUM 공개됨

---

## ⏳ 제출까지 남은 작업 (우선순위)

### 🔴 P0 — 필수 (4/29 까지)

- [ ] **표지 (00a_표지.md) 학번 기입** — 현재 `(학번 입력 필요)` placeholder
- [ ] **참고문헌 생성** — 현재 본문에 인용만 있고 별도 참고문헌 섹션 미확인 → 점검 + 필요시 작성
- [ ] **목차 (00d_목차.md) 자동 생성** — 페이지 번호 포함
- [ ] **통합본 최종 확정본 명명** — `박사논문_최종제출_YYYYMMDD.docx`
- [ ] **PDF 변환** — LibreOffice 또는 Word 로 docx → PDF (호서대 규격 여백 재확인)
- [ ] **페이지 수 검증** — 목표 100-120 페이지. 현재 예상 110-125.
- [ ] **지도교수 (문남미 교수) 최종 검토**

### 🟡 P1 — 권장 (여유 시)

- [ ] **그림 6-1 f1_char 0.000 표시 정리** — sentence 프롬프트에서 측정 안 된 값이 0으로 오해됨, 주석/바 제거 또는 N/A 로 처리
- [ ] **영어 합성 PPO 학습 (deferred)** — $5, ~1.5시간. 완료 시 Ch.6 §7.6 에 L-DWA English 수치 추가 가능
- [ ] **Ch.2 관련연구 확장** — RAG 연구 흐름 최신화 (2026 arXiv 인용 추가)
- [ ] **감사의 글** 작성

### 🟢 P2 — 포스트 제출 (5월 이후)

- [ ] 영어 합성 cache + 1-seed PPO + 수치 포함 논문 업데이트 (GitHub 공개 저장소에)
- [ ] 학술지 논문 (JKSCI 계열 또는 IEEE Access) 투고 준비
- [ ] BERT intent classifier 학습 + 재평가
- [ ] 선행 저장소 CORRIGENDUM 확장 (재측정 원 notebook 에 경고 주석 추가)

---

## 🔧 "지금 바로" 실행 가능 작업

사용자가 복귀했을 때 쉽게 이어갈 수 있는 구체적 명령:

### 1. Word 에서 v10 final 열기 + 검토
```
D:\triple_rag_phd\thesis_current\v5_rewrite\박사논문_통합본_v10_final.docx
```
2,117줄 → 약 110-120 페이지 예상. Word 에서 열어 페이지 수 확인.

### 2. 학번 기입
```
docs 편집 후 → python scripts/build_unified_thesis_v6.py 재실행하여 통합본 갱신
```

### 3. 참고문헌 점검
```powershell
# 본문의 인용 패턴 확인
grep -h "(\[.*\]\|et al\.,\?.*\d{4})" thesis_current/v5_rewrite/*.md
# → 현재 본문에 언급만 있고 참고문헌 섹션 부재 → 신규 작성 필요
```

### 4. PDF 생성 (컨테이너 내부)
```bash
docker-compose exec triple_rag bash
cd /tmp
libreoffice --headless --convert-to pdf /workspace/thesis_current/v5_rewrite/박사논문_통합본_v10_final.docx
```

### 5. Streamlit 대시보드 (심사 리허설용)
```bash
docker-compose exec triple_rag bash
cd /workspace
streamlit run streamlit_app/dashboard.py --server.port 8501 --server.address 0.0.0.0
# → http://localhost:8501
```

---

## 📞 사용자 복귀 시 Claude 에게 한 마디

**다음 단계**를 명확히 지시:

- `"참고문헌 점검부터"` → P0 항목
- `"PDF 만들어봐"` → docx → PDF 변환
- `"영어 PPO 돌려"` → P1 deferred 작업
- `"페이지 수 확인"` → docx 분량 계산
- `"감사의 글 초안"` → P1 보강

---

**마지막 업데이트**: 2026-04-22 22:00 KST (외출 중)
**다음 체크인**: 복귀 시 `gh pr list` 로 대기 중인 PR 확인
