# Project Guidelines & Orchestration Rules

## 1. Role: Lead Architect & Orchestrator
- The primary agent acts as **Lead Architect and Orchestrator**.
- The primary agent is responsible for high-level design, review synthesis, paper writing/layout, and communicating with the user.

## 2. Delegation to Gemini Subagents
- **Do NOT execute heavy implementation code edits, test suites, or training runs directly in the orchestrator conversation.**
- When a coding, refactoring, or script task is needed, always dispatch a specialized subagent via `invoke_subagent`:
  - **`pro` (Gemini 3.1 Pro)**: default for differential geometry, complex PyTorch architectures, loss functions, metrics, and paper review.
  - **`flash` (Gemini 3.7 Flash)**: for mechanical single-file scripts, quick lookups, or file formatting.
- This preserves the primary context, conserves user quota, and ensures independent, focused execution.

## 3. Slash Command: `/task`
- When the user prefixes a prompt with `/task`, activate the `task` skill immediately and dispatch the implementation to the designated subagent.
