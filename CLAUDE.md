# CLAUDE.md

`README.md` says what this is; `docs/REPORT.md` holds every result and how it was
measured. This file covers what breaks when you edit things.

## Commands

```bash
python3 prepare.py ~/Documents/ArcGIS/Projects/NorthSea/FELN.json data   # stdlib only, runs on the Mac
rsync -a prepare.py train.py predict.py app.py index.html data gc1:feln-unsloth/
# gc1 (~/feln-unsloth/.venv, unsloth 2026.10.3), always inside tmux session feln-unsloth:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --model unsloth/Qwen3.5-0.8B --out runs/<name> 2>&1 | tee logs/<name>.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --out runs/smoke --max-steps 20   # smoke test, about 2 min
.venv/bin/python predict.py runs/<name>/adapter "question"
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8095                   # tmux window web
```

Training, scoring and the web app need the GPU box; nothing here trains on the Mac.

## Invariants

- `prepare.QUESTIONS` (instructions + 12 option descriptions) is part of the model's
  input. `train.py`, `predict.py` and `app.py` all import it. Editing that text, the
  option keys or `label()` changes what every trained run sees at inference without
  retraining it. Retrain, or keep a copy of the old text for old runs.
- Labels are `primary+sorted(secondaries)`. Secondary order is deliberately not part
  of the label (FELN compare matches secondaries by name).
- The split is stratified by label with seed 3407. Changing the seed, the ratios or
  `FELN.json` changes `data/`, which makes new runs incomparable with `runs/*`.
  Regenerate and re-run all runs together. Test rows must never be used for
  calibration or model choice; calibration uses val.
- Laya (`convaiinnovations/laya`) needs `--no-4bit` and has no adapter save, only
  `merged/`. `predict.py` and `app.py` load `adapter/` in 4-bit and `merged/` in 16-bit.
- Report `test_breakdown.exact` (per-row `predict`), not `after.accuracy`
  (`evaluate`). They can differ by 1-3 questions (REPORT section 10).

## Results etiquette

One test question is 0.33 points, and seed noise is up to 1.7 points. Do not claim a
winner from differences under about 2 points. The two confident "misses" in REPORT
section 9 are label noise in `FELN.json`, which this repo does not modify. Keep them
in the numbers.

## gc1

`ssh gc1` (Ubuntu, 4x RTX PRO 6000 96 GB). Weights live only on gc1 under
`~/feln-unsloth/runs/<run>/{adapter,merged}` (about 70 GB total, not in git). The web
tester binds to 127.0.0.1; reach it with `ssh -fN -L 8095:127.0.0.1:8095 gc1`. Do not
bind it to 0.0.0.0: it has no auth.
