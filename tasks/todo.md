# feln-unsloth: decision model for FELN layer selection

Goal: given a NorthSea question, pick which 1, 2 or 3 layers (Wells, Pipelines,
Discoveries) the FELN query uses, and which one is primary, with calibrated
probabilities. Unsloth `FastDecisionModel` + `DecisionTrainer`, trained on gc1.

Data facts (`~/Documents/ArcGIS/Projects/NorthSea/FELN.json`, 3000 rows):
- 1 layer 498, 2 layers 1659, 3 layers 843. 15 ordered combos, 7 unordered sets.
- Keyword baseline: layer set right 43%, first-mention primary right 93%.
  Subtype words overlap (gas exists in all three layers), so text alone is
  sometimes ambiguous; calibrated probabilities matter.

## Plan

- [x] 1. Confirm Unsloth decision dataset row format (state / questions / gold)
       from `typed-decisions` and `FastDecisionModel.build_dataset` source.
- [x] 2. `prepare.py`: FELN.json -> decision rows. One `choice` question,
       12 options = primary + unordered secondaries
       (e.g. `Wells`, `Wells+Pipelines`, `Wells+Discoveries+Pipelines`).
       Stratified split 80/10/10 (train/val/test), seed 3407. Self-check asserts.
- [x] 3. Baselines on test: prior (majority) and keyword+first-mention.
- [x] 4. gc1: `~/feln-unsloth`, fresh uv venv, `pip install unsloth`; smoke
       run `max_steps=20` inside tmux session `feln-unsloth`.
- [x] 5. Full train in tmux: Qwen3.5-4B, 4-bit LoRA r16, lr 2e-4, 3 epochs
       (small data), calibrate on val, evaluate on test. Log to file.
- [x] 6. Save adapter + merged; pull metrics back; `predict` demo on a few
       hand-written questions.
- [x] 7. README with results (accuracy, ECE, per-count confusion) vs baselines.

## Review

- Baselines (test 300): prior 10.7%, keyword+first-mention 47.3%.
- Smoke 20 steps (99 s): before 5.7% -> after 76% exact, primary 99.7%.
- Full 225 steps (8 min): test 94.3% exact, count 100%, ECE 0.021; conf>=0.7 -> 98.5% acc at 89% coverage.
- 2 confident misses are FELN.json humanize label noise (swapped wells/pipelines; dropped layer word). Not fixed here.
- 0.8B x4 (one per GPU): r16 94.3% (= 4B), seed1 93.7%, r64 95.3%, 5ep 93.3% (overconfident). 12/17 misses shared with 4B; size is not the bottleneck.
- Round 3: gemma4-e4b 95.3%, qwen35-2b 95.0%, llama32-3b 94.3%, gemma4-e2b 91.7%, laya 90.3-92.0% (16-bit, 2 min). Laya save crash fixed (merged only). Docs: docs/REPORT.md.
- Web tester: app.py (FastAPI) + index.html (vanilla JS) on gc1 127.0.0.1:8095, tmux feln-unsloth:web, via SSH tunnel. ~85 ms/question after 14 s first load. Server and tunnel stopped 2026-10-09.
- Repo: github.com/mraad/feln-unsloth (private), work merged via PR.
