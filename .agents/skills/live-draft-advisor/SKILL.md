---
name: live-draft-advisor
description: Live fantasy football draft advisor. Whenever the user pastes draft results, player names, or screenshots, automatically parses the drafted players, updates draft state and user's roster, and generates the 3 best strategic contingencies for the next pick.
---

# Live Draft Advisor Skill

This skill is invoked during a live fantasy football draft whenever the user provides draft updates, pasted text from Yahoo, or drafted player names.

## Workflow

1. **Parse & Update**:
   Execute `live_draft_engine.py` using `.venv/bin/python` to update the draft state:
   ```bash
   .venv/bin/python src/live_draft_engine.py --input "<raw_pasted_text>"
   ```

2. **Output Format**:
   Always format the response with:
   - **Board Summary**: Current pick number, user's next pick, and distance until user's turn.
   - **Your Roster**: Current players drafted and positional breakdown (QB, RB, WR, TE).
   - **3 Strategic Contingencies**:
     1. **Contingency 1 (Primary Target)**: Best player / optimal positional fit based on NN VORP and roster need.
     2. **Contingency 2 (Positional Pivot)**: High-upside alternative if the primary target is sniped.
     3. **Contingency 3 (Safety / Positional Advantage)**: Fallback option or elite TE/QB pivot.
   - **Top Available Board**: Clean table of top available players at each key position (WR, RB, TE, QB) with NN projected points and VORP.
