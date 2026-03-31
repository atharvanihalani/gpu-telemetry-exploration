---
name: Multi-agent workflow preferences
description: Atharva likes parallel worktree agents for independent implementation tasks, with review-then-merge flow
type: feedback
---

Atharva is comfortable with multi-agent orchestration using worktree isolation for independent tasks. Validated workflow in session 3:

1. Atharva reviews plans one by one, asks clarifying questions
2. When satisfied, says "go" and agent is launched in a worktree
3. Multiple agents run in parallel (up to 3-4 at once worked fine)
4. Merge as they complete — copy files from worktree, verify import, commit to main
5. Batch push when convenient

**Why:** Atharva explicitly wanted to practice multi-agent orchestration. The pattern of "review plan → ask Qs → launch agent → merge" worked smoothly for 13 independent workload scripts in one session.

**How to apply:** For tasks that decompose into independent file-level units (new scripts, new modules), prefer this parallel agent pattern. Ask clarifying questions before launching each agent. Don't wait to batch — merge as they land.
