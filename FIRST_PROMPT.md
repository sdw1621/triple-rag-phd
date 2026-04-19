# 🎬 First Prompt for Claude Code

> Copy and paste the block below into your Claude Code terminal as your FIRST message.
> This gets Claude Code up to speed on everything.

---

## 📋 Step-by-Step Instructions

### 1. Setup before first prompt

```powershell
# Windows PowerShell
cd C:\Users\shin\triple-rag-phd

# Start Claude Code
claude
```

### 2. First prompt (copy everything in the ```text block below)

```text
Hi! I'm starting this PhD thesis project and handing off from Claude Desktop (web chat) to you (Claude Code) for the actual implementation work.

Please do the following in order:

1. Read `CLAUDE.md` in the project root — this has the full project context.

2. Read `context/CONVERSATION_HISTORY.md` — this summarizes the ~7 hours of prior work on Claude Desktop that led to this handoff. I don't want to re-explain everything.

3. Read `context/THESIS_CONTEXT.md` — this is the PhD thesis content summary. Your code must align with these thesis descriptions.

4. Read `context/PRIOR_WORK_ANALYSIS.md` — this explains how to port code from my previous JKSCI 2025 paper repo.

5. Read `context/CODE_SPECS.md` — this is the detailed implementation spec for every module.

6. Read `context/ROADMAP.md` — this is the task-by-task work plan.

7. After reading everything, please report back with:
   - A 1-paragraph summary showing you understood the project
   - Current environment status (Docker container, Python version, CUDA, etc.)
   - What you think the next 3 immediate tasks are
   - Any questions or concerns before we start

Do NOT start implementing anything yet. Just read, understand, and report back.

The thesis submission deadline is April 30, 2026. Today is April 19, 2026. We have 11 days.
```

### 3. Evaluate Claude Code's response

After Claude Code responds:

- **If it correctly summarizes the project** and proposes reasonable next tasks → proceed with T1.9 (Docker rebuild)
- **If it misses key context** → point to specific files in `context/` folder
- **If it asks good questions** → answer them before proceeding

### 4. Second prompt (start work)

Once Claude Code is ready:

```text
Great, let's start. Please proceed with:

1. T1.9 — Rebuild Docker container with numpy<2.0 fix
   - docker-compose down
   - docker-compose build --no-cache
   - docker-compose up -d

2. T1.10 — Smoke test inside container
   - Verify PyTorch 2.1.2, NumPy 1.x, CUDA available

3. Report back with results before proceeding.
```

---

## 💬 Useful Recurring Prompts

### Daily start (when resuming work)

```text
Good morning! Please:
1. Check git status
2. Read context/ROADMAP.md for current progress
3. Give me today's standup report in the format described in ROADMAP.md
4. Propose what to work on today
```

### When starting a new module

```text
Let's implement [MODULE_NAME] according to context/CODE_SPECS.md.

Prerequisites:
- Read the module spec
- Check if prior repo has a reference implementation
- Plan the file structure

Then:
1. Create the file
2. Implement per spec
3. Write tests
4. Run tests
5. Show me the test results before committing
```

### After completing a task

```text
Great work! Please:
1. Update context/ROADMAP.md (check off the completed task)
2. Git commit with conventional message
3. Git push
4. Suggest next task
```

### When something breaks

```text
We have an issue. Please:
1. Show me the exact error
2. Show me what command caused it
3. Check docker-compose logs
4. Propose 2-3 possible fixes
5. Don't apply fixes yet, let me choose
```

### Weekly milestone review

```text
End-of-week review:
1. List all commits this week (git log --since=7.days)
2. Compare to ROADMAP.md milestones
3. Tell me if we're on track for 4/30 submission
4. Identify any risks
```

---

## ⚠️ Important Notes for the Author

### When NOT to use Claude Code

Some tasks are easier on Claude Desktop:
- Writing thesis prose
- Generating figures (matplotlib with complex styling)
- Discussing research strategy
- Reviewing drafts for readability

### When to use Claude Code

Code-centric tasks:
- Implementing modules
- Running tests
- Git operations
- Docker management
- Debugging code
- Refactoring

### Bridge pattern

You (the author) are the **bridge** between Claude Desktop and Claude Code:

1. **Claude Desktop**: "We should implement feature X with approach Y"
2. You: Copy the approach to a new message in Claude Code
3. **Claude Code**: Implements, tests, commits
4. You: Copy results back to Claude Desktop for review
5. **Claude Desktop**: Proposes refinements
6. Repeat

### Keeping both in sync

- After major changes, update `CLAUDE.md` and `context/CONVERSATION_HISTORY.md`
- Both Claude Desktop and Claude Code should be shown the latest state
- Git is the ultimate source of truth

---

## 🎯 Success Metrics

By end of Day 1 (2026-04-19):
- [ ] Repo on GitHub, first commit pushed
- [ ] Docker rebuilt with numpy<2.0
- [ ] Data downloaded (prior repo + 3 benchmarks)
- [ ] Claude Code successfully loaded CLAUDE.md

By end of Day 2-3 (4/20-21):
- [ ] Core RAG modules ported (vector, graph, ontology, rdwa, metrics)
- [ ] Integration test: end-to-end query returns answer
- [ ] ~20 commits to GitHub

By end of Day 5 (4/23):
- [ ] BERT intent classifier working
- [ ] Offline cache design + small-scale test

By end of Day 6 (4/24):
- [ ] PPO modules implemented
- [ ] Cache build started (runs overnight)

By end of weekend (4/27):
- [ ] All experiments run
- [ ] Actual numbers ready for thesis

By 4/30:
- [ ] 🎓 THESIS SUBMITTED

Good luck! 💪
