<!-- Copyright 2026 Sean M. Stow. All rights reserved. -->
# ML Laboratory — Multi-Model Collaboration Build Plan

## Problem (from user)
In the chat box, loaded models give the same output and don't work together — they
wait for each other instead of running simultaneously. Want divergence + simultaneity
at speed.

## What already exists in /home/daddy/Documents/lm_lab/ML-Labrotory/
- Per-slot ModelRole (General/Coder/Researcher/Critic/Planner/Writer/Custom) with
  system_prompt() — already drives per-slot system prompts (app.rs compose_system_prompt).
- Multi-model slots, split_chat_view, dual_run_mode, "Send to Both (Enter)", "Ask all".
- Broadcast: fires send_prompt on every slot via rt.spawn — tasks are spawned, not
  blocked, in poll_messages (app.rs ~line 1394-1414).
- Relay (sequential pipeline), Synthesize (merge all latest answers into focused slot),
  Compare tab (side-by-side latest answers).
- Speed work already done: optimal_threads, mmap, num_batch, tcp_nodelay, warm_model,
  stream segment caching, idle vs 60fps throttle.

## Root causes to fix
1. Same output:
   - If every slot has the same model + same role (esp. General), prompt is effectively
     identical → near-identical output. Roles exist but aren't pushed as the divergence lever.
   - No per-prompt angle injection: all slots get the same user text + same role prompt.
2. Waiting for each other (feels sequential):
   - Split-view send guard: Enter disabled while ANY slot is streaming
     (app.rs ~line 3233: `!active_slots.iter().any(|&s| self.slots[s].chat.is_streaming())`).
     That blocks a second send to idle slots while one is working.
   - If split_chat_view is off or active_slots < 2, user is in single-slot mode and
     never sees parallelism. Tooling must make the multi-model path obvious.

## Build order
### 1. Per-slot send readiness (free parallel sends to idle slots)
- Allow "Send to Both"/per-slot sends to idle slots even while other slots stream.
- Keep a per-slot guard (slot.is_streaming), not the global any-streaming guard, for
  the broadcast path. The focused single-slot path still blocks its own slot while
  streaming (that's correct).
- Show per-column streaming state prominently so the user can see who's busy.

### 2. Make roles the divergence lever (UI + safeguards)
- In the split-view multi-model toolbar and slot cards, surface each slot's role +
  model together, not just model.
- Add a same-model + same-role warning when 2+ active slots share model AND role,
  with a one-click "diversify" action (swap role on one slot, or pick a different model).
- Default the "Team" presets to genuinely different roles (Planner/Coder/Critic etc.).

### 3. Per-prompt contrast suffix (new small feature)
- A toggle in the unified input dock: "Contrast angles" on/off.
- When on and dual_run_mode is on, prepend a per-slot angle instruction to the prompt
  before sending, so the same model family still answers from different angles.
- Angles rotate by slot role: Direct answer / Critique / Trade-offs / Step-by-step /
  Alternatives. Stored as a small map Role -> suffix.

### 4. Labeled "Team" flow (wire existing pieces)
- Add a "Team" button in the chat toolbar: Broadcast (parallel, all active slots) then
  surface a Synthesize action once all have answered, optionally auto-advance to Compare.
- This is the "work together" flow: diverge in parallel, then merge.

### 5. Concurrency hygiene for shared large models
- When the same model is assigned to 2+ active slots, show a small note: "Same model on
  N slots — concurrent streams share the model's inference budget."
- Not a hard block — just honest feedback so the user knows why two same-model streams
  might not be 2x faster.

## Files to patch
- src/ui/app.rs — split-view send guards, toolbar role visibility, "Team" button,
  same-model warning, contrast-toggle wiring.
- src/ui/chat.rs — contrast suffix support in send_prompt / send_message signature path
  (keep backward compat; add optional per-slot angle injection).
- Possibly src/ui/relay.rs / src/commands.rs — expose "Team" as a command if desired.

## Verification
- `cargo check` after each patch (fast, no codegen).
- `cargo test` before and after to keep the 95+ test bar intact.
- Release build only after the logic shape is settled.

## Out of scope for now
- Full agent framework rewrite — not needed; the scaffolding is already here.
- Remote/cloud models — stays loopback-only per the security posture.
