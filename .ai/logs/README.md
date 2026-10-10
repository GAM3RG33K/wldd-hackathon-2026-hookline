# Logs — progress + verification trail

One file per work session: `.ai/logs/YYYY-MM-DD-<topic>.md`.
Append, never rewrite history. Each entry: goal → commands run → outputs → verdict → next.

## Template
```md
# YYYY-MM-DD — <topic>
- Goal:
- Commands: (exact, with workdir)
- Result: (pass/fail + artifact paths)
- Verdict: (what this proves / blocks)
- Next:
```
## Rules
- Log every verification-loop run (AGENTS.md) here, not in chat only.
- Reference artifacts by path (`out/<stem>/report.json`). No large pastes.
- Open blockers stay in the newest log until closed; close with date + commit/proof.
