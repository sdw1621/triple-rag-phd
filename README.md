# 📦 Handoff Package — How to Use

> This package transfers all Claude Desktop conversation context to Claude Code.

---

## 🎯 What's Inside

```
claude-code-handoff.zip
├── README.md                       # ← You are reading this
├── CLAUDE.md                       # Claude Code auto-loads this
├── FIRST_PROMPT.md                 # Copy-paste prompts for Claude Code
└── context/
    ├── CONVERSATION_HISTORY.md     # Summary of 7 hours of Claude Desktop work
    ├── THESIS_CONTEXT.md           # PhD thesis chapter contents
    ├── PRIOR_WORK_ANALYSIS.md      # Prior repo (JKSCI 2025) mapping
    ├── CODE_SPECS.md               # Module implementation specs
    └── ROADMAP.md                  # Task-by-task schedule to 4/30
```

---

## 📊 그림 재생성 (`scripts/plot_figure*.py`)

### ⚠️ 먼저 한글 폰트를 준비하세요

`plot_figure3_1`, `4_1`, `5_1`, `6_3`, `6_4`, `6_5` 는 한글 라벨을 사용하며
`cache/malgun.ttf` 를 찾습니다. **`cache/` 는 `.gitignore` 대상이라 clone 직후에는
없습니다.** 준비하지 않고 실행하면 그림의 한글이 **모두 두부(□)로 깨진 채** 저장됩니다.

```powershell
# Windows
copy C:\Windows\Fonts\malgun.ttf cache\malgun.ttf
```

폰트가 없으면 스크립트가 `RuntimeWarning` 으로 경고합니다. 경고를 보면 위 명령을
먼저 실행하세요.

### 용어: A-DWA

논문 최종본은 알고리즘 명칭을 **A-DWA**(Adaptive Dynamic Weighting Algorithm)로
확정했습니다. plot 스크립트와 `docs/figures/` 는 A-DWA 로 통일되어 있습니다.
단, 결과 파일명(`rerun_ldwa_seed42_*.json`, `ldwa_cache_ppo_*.pt`)은 실제 경로이므로
`ldwa` 를 그대로 씁니다. `thesis_current/` 원고는 이전 명칭(L-DWA)이며
이력 자료입니다 — `thesis_current/README.md` 참조.

### `docs/figures/` 가 authoritative

`docs/figures/*.png` 는 **논문에 실제로 수록된 그림**입니다. 일부 스크립트는
수록본과 라벨 배치가 완전히 일치하지 않으므로(특히 `plot_figure4_1` — 수록본에는
지시선이 있으나 스크립트에는 없음), 스크립트 출력을 그대로 논문에 쓰지 말고
수록본과 대조하세요.

---

## 🚀 Installation (3 minutes)

### Prerequisites
- `triple-rag-phd-initial-setup.zip` already extracted to `C:\Users\shin\triple-rag-phd\`
- GitHub repo `sdw1621/triple-rag-phd` created (Private)

### Step 1: Extract this package

**Windows PowerShell**:
```powershell
# Navigate to project folder
cd C:\Users\shin\triple-rag-phd

# Extract handoff package into project
# (CLAUDE.md goes to root, context/ folder goes to ./context/)
Expand-Archive claude-code-handoff.zip -DestinationPath . -Force

# Verify
dir
dir context
```

**Expected after extraction**:
```
C:\Users\shin\triple-rag-phd\
├── CLAUDE.md                   ← NEW (handoff)
├── FIRST_PROMPT.md             ← NEW (handoff)
├── README.md                   ← from init ZIP
├── Dockerfile                  ← from init ZIP
├── docker-compose.yml          ← from init ZIP
├── context/                    ← NEW (handoff folder)
│   ├── CONVERSATION_HISTORY.md
│   ├── THESIS_CONTEXT.md
│   ├── PRIOR_WORK_ANALYSIS.md
│   ├── CODE_SPECS.md
│   └── ROADMAP.md
├── docs/                       ← from init ZIP
│   └── PROJECT_HISTORY.md
├── data/, src/, tests/, ...    ← from init ZIP
└── ...
```

### Step 2: Commit to Git

```powershell
cd C:\Users\shin\triple-rag-phd

git add CLAUDE.md FIRST_PROMPT.md context/
git commit -m "docs: add handoff package from Claude Desktop

Includes:
- CLAUDE.md: Project context auto-loaded by Claude Code
- FIRST_PROMPT.md: Initial prompt template
- context/CONVERSATION_HISTORY.md: Summary of prior Desktop sessions
- context/THESIS_CONTEXT.md: PhD thesis chapter contents
- context/PRIOR_WORK_ANALYSIS.md: Prior repo migration plan
- context/CODE_SPECS.md: Module implementation specifications
- context/ROADMAP.md: 11-day task schedule to submission"

git push
```

### Step 3: Install Claude Code (if not done)

```powershell
# Native installer (recommended, no Node.js needed)
irm https://claude.ai/install.ps1 | iex

# Verify (open NEW PowerShell window)
claude --version
```

### Step 4: Start Claude Code

```powershell
cd C:\Users\shin\triple-rag-phd
claude
```

First time authentication:
- Browser opens automatically
- Login with Claude.ai account (Pro/Max subscription needed)
- Approve access
- Return to terminal

### Step 5: Send the first prompt

Open `FIRST_PROMPT.md` in a text editor, copy the prompt in the code block, paste into Claude Code.

---

## ✅ Verification Checklist

After installation:

- [ ] `CLAUDE.md` exists at project root
- [ ] `context/` folder has 5 files
- [ ] `FIRST_PROMPT.md` exists at project root
- [ ] Git shows all handoff files committed
- [ ] GitHub has latest commit
- [ ] Claude Code launches successfully
- [ ] First prompt received a comprehensive response
- [ ] Claude Code understands project context (no repeated questions)

---

## 🎬 Next Steps

Once Claude Code responds to the first prompt:

1. **Verify understanding**: Make sure Claude Code grasped the project
2. **Start T1.9**: Rebuild Docker (numpy<2.0 fix)
3. **Start T2.1**: Download data from prior repo
4. **Start T2.2**: Port vector_store.py

See `context/ROADMAP.md` for full task list.

---

## 🔄 Ongoing Sync Between Claude Desktop and Claude Code

### When Claude Desktop does something important

Example: "Claude Desktop generated a new figure"

Update `context/CONVERSATION_HISTORY.md`:
```markdown
### Update 2026-04-20
- Claude Desktop generated Figure 5-5 (PPO architecture)
- Saved to docs/figures/fig_5_5_ppo_architecture.png
- Thesis Ch.5 updated to reference this figure
```

Commit and push. Claude Code will pick up changes on next session.

### When Claude Code does something important

Example: "Claude Code completed PPO trainer"

Claude Code should auto-update:
- `context/ROADMAP.md` (check off T4.3.*)
- Git commit with descriptive message
- Tag if milestone completed

The author can then share the git log with Claude Desktop for review.

---

## 🆘 Troubleshooting

### Q: Claude Code says it doesn't see CLAUDE.md
- **Check**: `ls CLAUDE.md` in terminal
- **Fix**: Make sure extraction was to project root (not nested folder)

### Q: Claude Code asks questions already answered in context/
- **Check**: Did it actually read the files? Ask: "Did you read context/CONVERSATION_HISTORY.md?"
- **Fix**: Explicitly ask to read each file

### Q: PowerShell Expand-Archive overwrites existing files
- **Expected**: Use `-Force` flag, it only overwrites same-name files
- **Safety**: The handoff ZIP only has CLAUDE.md, FIRST_PROMPT.md, context/ — no conflicts with init ZIP

### Q: Context is too large, Claude Code hits token limit
- **Split**: Tell Claude Code to read files one at a time, summarize each before moving on
- **Priority order**: CLAUDE.md > CONVERSATION_HISTORY.md > ROADMAP.md > CODE_SPECS.md > THESIS_CONTEXT.md > PRIOR_WORK_ANALYSIS.md

---

## 📝 Package Contents Summary

| File | Lines | Purpose |
|---|---|---|
| CLAUDE.md | ~300 | Always-loaded project context |
| FIRST_PROMPT.md | ~200 | Onboarding prompt templates |
| context/CONVERSATION_HISTORY.md | ~280 | What happened before handoff |
| context/THESIS_CONTEXT.md | ~250 | Thesis chapter contents |
| context/PRIOR_WORK_ANALYSIS.md | ~280 | Prior repo migration |
| context/CODE_SPECS.md | ~600 | Module specifications |
| context/ROADMAP.md | ~320 | 11-day task schedule |
| **Total** | **~2,230** | **Complete handoff** |

---

**Created**: 2026-04-19 by Claude Desktop
**Purpose**: Seamless handoff to Claude Code for implementation phase
**Deadline**: 2026-04-30 PhD thesis submission
