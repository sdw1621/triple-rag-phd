# 🌅 WAKEUP — M8 스모크 테스트 재개 가이드

**작성**: 2026-04-22 (Claude가 사용자 수면 중 자동 진행 중 중단)

---

## 📌 현재 상태

### ✅ 자동으로 완료된 것

1. **M8 reproducibility defense 커밋** — cherry-pick 깔끔히 완료
   - 브랜치: `claude/m8-list-prompt-f1char` (메인 저장소 `D:\triple_rag_phd`)
   - 커밋: `c6c8c91 feat(M8): reproducibility defense — list prompt + f1_char + triple F1 view`
   - 코드 변경:
     - `src/eval/metrics.py` — `f1_char` 추가 (char-3gram F1)
     - `src/rag/triple_hybrid_rag.py` — `PROMPT_TEMPLATE_LIST` 추가
     - `scripts/evaluate_rerun.py` — `--prompt-style {sentence,list}` 플래그
     - `tests/test_eval_metrics.py` — 5개 테스트 추가
   - 저자의 `dff7dc1` 쉼표 버그 수정과 **충돌 없음** (다른 부분 수정)

2. **M8 방어 문서 업데이트** — 저자의 punctuation-fix 반영
   - `docs/M8_REPRODUCIBILITY_DEFENSE.md`에 **2-레이어 진단** 추가
   - Layer 1 (쉼표 버그, 저자가 이미 수정): F1_strict 0.072 → 0.137
   - Layer 2 (format mismatch, 이 PR이 제안): list prompt로 1.0까지 회복 가능

### 🔴 중단된 이유

**Docker Desktop 크래시** (04:54 스크린샷 확인):
```
An unexpected error occurred
starting services: initializing Inference manager:
remove C:\Users\sdw19\AppData\Local\Docker\run\dockerInference:
The file cannot be accessed by the system.
```

- Docker Desktop 프로세스 전체 종료 완료
- 문제 파일 삭제 시도 → **권한 불가** (named pipe, 시스템 잠금)
- `com.docker.service` Windows 서비스 재시작 시도 → **관리자 권한 필요**
- **사용자 수동 개입 필요**

---

## 🚀 깨어났을 때 재개 (3단계)

### ① Docker 복구 (택 1)

**가장 확실한 방법 — 윈도우 재부팅**:
```
재부팅 후 Docker Desktop 자동 시작 확인
```

**재부팅 없이 빠르게**:
```powershell
# PowerShell 관리자 권한으로 실행
wsl --shutdown
Remove-Item 'C:\Users\sdw19\AppData\Local\Docker\run\dockerInference' -Force
Start-Service com.docker.service
& 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
```

**또는** Docker Desktop 우클릭 → "Run as Administrator"

### ② Docker 준비 확인

```powershell
docker ps
# 빈 목록이 나오면 OK. 에러면 위 복구 반복.
```

### ③ 스모크 테스트 실행 (2분, $0.2)

```powershell
cd D:\triple_rag_phd
git checkout claude/m8-list-prompt-f1char
docker-compose up -d
docker-compose exec triple_rag bash
```

컨테이너 내부:
```bash
cd /workspace

# A. 센텐스 프롬프트 (기존) 100 QA — baseline 확인
python scripts/evaluate_rerun.py \
  --qa data/university/gold_qa_5000.json \
  --policy rdwa \
  --output results/smoke_rdwa_sentence.json \
  --prompt-style sentence \
  --limit 100 --workers 10

# B. 리스트 프롬프트 (신규) 100 QA — F1_strict 상승 여부 확인
python scripts/evaluate_rerun.py \
  --qa data/university/gold_qa_5000.json \
  --policy rdwa \
  --output results/smoke_rdwa_list.json \
  --prompt-style list \
  --limit 100 --workers 10
```

**통과 기준**:
- `smoke_rdwa_sentence.json`의 F1_strict ≈ 0.137 근처 (baseline 재확인)
- `smoke_rdwa_list.json`의 F1_strict **≥ 0.5** (크게 상승)

### ④ 전체 재측정 (통과 시만, 2시간, ~$10)

```bash
# R-DWA + Oracle + L-DWA(3 seeds) × list prompt × 5000 QA
# 상세 명령: docs/M8_REPRODUCIBILITY_DEFENSE.md § "Full rerun plan" 참고
```

---

## 🎯 판단 기준 (결과 나왔을 때)

| 결과 | 해석 | 다음 행동 |
|---|---|---|
| list F1_strict ≥ 0.5 | format 가설 맞음 | ✅ 전체 재측정 → Ch.6 업데이트 |
| list F1_strict 0.2~0.5 | 부분 개선 | ⚠️ 프롬프트 추가 튜닝 or f1_char 중심 전환 |
| list F1_strict < 0.2 | 가설 기각 | ❌ sentence baseline 유지, 현재 0.167로 논문 정리 |

**안전망**: 스모크 실패해도 **현재 F1_strict 0.167은 충분히 방어 가능**
(L-DWA vs R-DWA +21.9%, Oracle 99.4%, 3-seed std 0.002 — `CURRENT_STATUS_KR.md` §4 주장들).

---

## 📂 관련 파일

- `docs/M8_REPRODUCIBILITY_DEFENSE.md` — 전체 방어 전략 (상세)
- `results/CURRENT_STATUS_KR.md` — 4/21 기준 전체 상태
- `results/rerun_rdwa_fixed.json` — 저자가 이미 측정한 R-DWA (0.137)
- `results/rerun_ldwa_seed42_fixed.json` — L-DWA 측정값 (0.167)

---

## 💬 재개 시 Claude에게 한마디

> "스모크 테스트 진행해 (Docker 복구 완료)"

→ 바로 단계 ③부터 실행.
