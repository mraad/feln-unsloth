# FELN layer-selection decision model: report

Date: 2026-10-09. Machine: gc1 (AWS, 4x NVIDIA RTX PRO 6000 Blackwell, 96 GB each).
Repository: <https://github.com/mraad/feln-unsloth>.
Method: Unsloth decision models, following
[Train your own Decision Model with Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth).

## 1. Goal

A FELN query (`feln` project) names one, two or three map layers. `layers[0]` is the
primary layer, the one whose features are returned. Any other layers only filter it
through a spatial relation. Before generating the query, we want a small model that
reads a North Sea question and decides:

- how many layers it needs (1, 2 or 3),
- which layers they are (Wells, Pipelines, Discoveries),
- which one is primary,

with a probability we can trust, so a low-confidence answer can be routed elsewhere.

Unsloth's decision models do this without generating text. The LLM reads the input and
the option list once, a small head (Cloudflare Clef design) scores every option, and
Unsloth then fits a temperature on held-out data so the probabilities match how often
the model is right.

## 2. Source data

`~/Documents/ArcGIS/Projects/NorthSea/FELN.json`: 3,000 records `{text, meta, source_text}`.
`meta.layers` is the ordered layer list, `text` the humanized question (paraphrased
locally by `feln/scripts/humanize.py`), `source_text` the generator sentence.

| Layers per query | Rows |
|---|---|
| 1 | 498 |
| 2 | 1,659 |
| 3 | 843 |

There are three layers (Wells 2,184 mentions, Pipelines 2,099, Discoveries 2,062),
15 distinct ordered combinations and 7 distinct sets. No question text is duplicated.

The task is harder than spotting layer names. Questions often name a layer only by a
subtype word, and the subtype vocabularies overlap (`Layers.json` subtype coded values):

| Layer | Subtype column | Values |
|---|---|---|
| Wells | `content_type` | dry, oil, gas, oil shows, oil/gas, gas/condensate, gas shows, shows, ..., water, salt |
| Pipelines | `PipelinesType` | condensate, gas, injection, oil, unknown |
| Discoveries | `discovery_type` | gas, gas/condensate, oil, oil/gas, condensate, unknown |

"Find shows within 3 kilometers of condensate" names Wells by subtype ("shows"), but
"condensate" could be a pipeline or a discovery.

## 3. Decision design

One `choice` question named `layers` with **12 options**: the primary layer plus the
*unordered* set of secondary layers.

```
Wells                      Pipelines                      Discoveries
Wells+Pipelines            Pipelines+Wells                Discoveries+Wells
Wells+Discoveries          Pipelines+Discoveries          Discoveries+Pipelines
Wells+Discoveries+Pipelines  Pipelines+Discoveries+Wells  Discoveries+Pipelines+Wells
```

Why this shape (chosen over 15 ordered combos and over separate count / primary /
per-layer yes-no questions):

- One answer is always self-consistent. Separate questions could say "2 layers" and
  then mark three layers present.
- Secondary order is not part of the label. `feln`'s `FELNCompare` matches secondaries
  by name, so a permutation scores the same; the order in `FELN.json` mostly follows
  the order of the text.
- The count, the set and the primary can all be read off the label.

Each row, in Unsloth's decision dataset format (`state`, `questions`, `gold`; the
format of [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)):

```json
{
  "state": "Locate injection pipelines that cross any gas discoveries.",
  "questions": {"layers": {
    "type": "choice",
    "instructions": "Which North Sea layers does this map query need? The first layer is the one whose features are returned; any other layers only filter it by distance or containment. Wells are typed dry, oil, gas, ... Pipelines are typed ... Discoveries are typed ..., and have field labels.",
    "criteria": {
      "Wells": "Return wells only; no other layer is involved.",
      "Wells+Pipelines": "Return wells, constrained by pipelines.",
      "...": "..."
    }
  }},
  "gold": {"layers": "Pipelines+Discoveries"}
}
```

The instructions list each layer's subtype words, because the decision head sees the
instructions and option descriptions as part of its input.

## 4. Split and baselines

`prepare.py` builds the rows and splits them **stratified by label, 80/10/10, seed 3407**:
train 2,400, val 300, test 300. The splits share no question text (asserted).

- **train**: fine-tuning.
- **val**: per-epoch eval loss, then `FastDecisionModel.calibrate` fits the temperature.
- **test**: touched only for the before/after numbers below.

| Baseline (test) | Exact |
|---|---|
| Uniform chance (1/12) | 8.3% |
| Prior: always the most common train label | 10.7% |
| Keyword: layer names found in the text, first mention is primary | 47.3% |

The keyword rule finds the primary layer 93% of the time but the full layer set only
43% of the time on the whole file, because of the subtype words.

## 5. Environment on gc1

| | |
|---|---|
| Code | `~/feln-unsloth` (rsync of this folder) |
| Python env | `~/feln-unsloth/.venv`, `uv venv -p 3.12`, `uv pip install --upgrade unsloth datasets` |
| Versions | unsloth 2026.10.3, unsloth-zoo 2026.10.3, torch 2.14.1+cu130, transformers 5.17.0, trl 1.13.0; fastapi 0.143.0, uvicorn 0.54.0 (web tester) |
| tmux | session `feln-unsloth`, one window per run (`setup`, `smoke`, `full`, `q08-*`, `gemma4-*`, `llama32-3b`, `qwen35-2b`, `laya`) plus `web` (web tester, now stopped) |
| Logs | `~/feln-unsloth/logs/<run>.log` (each ends with `EXIT_CODE n`) |
| Outputs | `~/feln-unsloth/runs/<run>/` |

Every training job ran inside tmux, so a dropped SSH connection does not stop it:
`ssh gc1`, then `tmux attach -t feln-unsloth`. This environment is separate from
`~/Automodel/.venv` used by `feln-lora`, and none of `feln-lora`'s Blackwell workarounds
(`LD_LIBRARY_PATH`, sdpa) were needed.

## 6. Training recipe

`train.py` follows the Unsloth guide's "Train with code" example:

```python
model, tokenizer = FastDecisionModel.from_pretrained(model_name, max_seq_length=1024, load_in_4bit=True)
model = FastDecisionModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0,
                                         use_gradient_checkpointing="unsloth", random_state=3407)
items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)   # 0 rows skipped
trainer = DecisionTrainer(model=model, processing_class=tokenizer,
                          train_dataset=train_items, eval_dataset=val_items, args=TrainingArguments(
    per_device_train_batch_size=8, gradient_accumulation_steps=4, num_train_epochs=3,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10, weight_decay=0.01,
    bf16=True, eval_strategy="epoch", logging_steps=10, save_strategy="no", seed=3407))
trainer.train()
FastDecisionModel.calibrate(model, tokenizer, val_items)    # temperature fitted on val
FastDecisionModel.evaluate(model, tokenizer, test_items)    # also run before training
FastDecisionModel.predict(model, tokenizer, text, QUESTIONS) # per test row -> test_predictions.jsonl
model.save_pretrained(".../adapter"); model.save_pretrained_merged(".../merged")
```

Differences from the guide, and why:

- **3 epochs, not 2**: the guide suggests 3-4 for small data, and we have 2,400 rows.
  At batch 8 x accumulation 4 that is 225 steps.
- **Our own split instead of `split_holdout`**: a stratified val set for calibration
  and a separate test set no step looks at.
- **`max_seq_length=1024`**: a row is about 400 tokens; nothing was truncated.
- **Laya** loads in 16-bit (`--no-4bit`). Unsloth raises
  `NotImplementedError: Laya decision models train in 16-bit` for 4-bit. Laya also has
  no `save_pretrained` (adapter), only `save_pretrained_merged`.
- The head learning rate stays at the default 1e-4.

## 7. Results

All runs use the same data, split, seed (unless noted) and recipe. They are sorted by
test exact accuracy. **Exact** means the primary layer and the full layer set are both
right. ECE is the calibration error after calibrating on val. "Gate" keeps answers
with confidence ≥ 0.7.

| Run | Base model | Before | **Exact** | Set | Count | Primary | 2-layer | ECE | Gate: kept / right | Train time | Merged size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemma4-e4b` | gemma-4-E4B-it | 6.7% | **95.3%** | 95.7% | 100% | 99.7% | 91.6% | 0.030 | 93% / 98.2% | 10.7 min | 15 GB |
| `q08-r64` | Qwen3.5-0.8B, LoRA r64 | 4.7% | **95.3%** | 95.7% | 100% | 99.3% | 91.6% | 0.025 | 88% / 98.9% | 4.5 min | 1.7 GB |
| `qwen35-2b` | Qwen3.5-2B | 11.3% | **95.0%** | 95.3% | 99.7% | 99.3% | 91.6% | 0.032 | 90% / 98.5% | 5.2 min | 4.4 GB |
| `qwen35-4b` | Qwen3.5-4B | 5.7% | 94.3% | 94.7% | 100% | 99.3% | 89.8% | 0.021 | 89% / 98.5% | 8.2 min | 8.8 GB |
| `q08-r16` | Qwen3.5-0.8B | 4.7% | 94.3% | 94.7% | 100% | 99.3% | 89.8% | **0.018** | 89% / 98.5% | 4.9 min | 1.7 GB |
| `llama32-3b` | Llama-3.2-3B-Instruct | 8.7% | 94.3% | 94.7% | 99.7% | 99.3% | 90.4% | 0.034 | 89% / 98.5% | 5.6 min | 6.3 GB |
| `q08-r16-seed1` | Qwen3.5-0.8B, seed 1 | 4.7% | 93.7% | 94.0% | 99.7% | 99.3% | 89.2% | 0.019 | 88% / 98.1% | 4.1 min | 1.7 GB |
| `q08-r16-5ep` | Qwen3.5-0.8B, 5 epochs | 4.7% | 93.3% | 93.7% | 100% | 99.3% | 88.0% | 0.029 | 93% / 96.4% | 6.8 min | 1.7 GB |
| `laya-nosave` | Laya (ModernBERT), 16-bit | 8.3% | 92.0% | 92.3% | 98.3% | 99.7% | 88.6% | 0.041 | 88% / 94.7% | 1.8 min | (not saved) |
| `gemma4-e2b` | gemma-4-E2B-it | 12.7% | 91.7% | 92.0% | 99.7% | 99.3% | 85.6% | 0.037 | 91% / 97.1% | 7.6 min | 9.7 GB |
| `laya` | Laya (ModernBERT), 16-bit | 8.3% | 90.3% | 90.3% | 96.0% | 99.7% | 88.6% | 0.020 | 85% / 96.5% | 2.1 min | 808 MB |
| `smoke` | Qwen3.5-4B, 20 steps | 5.7% | 76.0% | 76.3% | 83.3% | 99.7% | 71.9% | 0.053 | – | 1.7 min | – |

Exact by layer count for the best runs: 1 layer 100%, 3 layers 100%. **All remaining
errors are in two-layer questions.**

"Before" is the untrained decision head on the test set, so chance level. For Laya it
is the published checkpoint used zero-shot on this task (8.3%, and badly overconfident:
ECE 0.52).

Train times are wall time for `trainer.train()` only. The four 0.8B runs ran at the
same time, one per GPU. Laya shared GPU 3 with `qwen35-2b`, so those two times are
slightly inflated. Peak GPU memory was not recorded; the 0.8B runs used 2-5 GB.

### Reading the numbers

- **One test question = 0.33 points.** Changing only the seed (`q08-r16` vs
  `q08-r16-seed1`) moved exact accuracy 0.6 points, and Laya's two identical runs
  differ by 1.7 points. Every run from 93.7% to 95.3% is within that noise. Model
  size barely matters here: the 0.8B matches the 4B.
- **Gemma 4 E4B, Qwen3.5-2B and Qwen3.5-0.8B r64 are tied at the top** (14-15 misses).
  The 0.8B is 9x smaller than the E4B and trains in less than half the time.
- **Laya is about 3-4 points behind the LLMs, but it is the smallest and fastest**:
  ModernBERT, 808 MB merged, about 2 minutes to train. Of all models, it made the most
  layer-count errors (96-98% vs 100%).
- **More epochs hurt calibration.** At 5 epochs the 0.8B kept more answers above the
  0.7 gate (93%), but fewer of them were right (96.4%).
- **An ensemble helps a little.** Averaging the probabilities of `q08-r16`,
  `qwen35-2b`, `qwen35-4b`, `llama32-3b` and `gemma4-e4b` gives 96.0% exact (computed
  from the saved predictions; not packaged as a model).

## 8. Confidence

Calibration is fitted on val and checked on test. ECE is 0.018-0.041 after
calibration for every run. The probabilities can be used as a gate. For `q08-r16`:

| Keep answers with confidence ≥ | Kept | Right |
|---|---|---|
| 0.5 | 100% | 94.3% |
| 0.7 | 89% | 98.5% |
| 0.8 | 85% | 99.2% |
| 0.9 | 82% | 99.2% |

Hand-written questions (`predict.py`):

| Question | `q08-r16` | `laya` |
|---|---|---|
| Find oil discoveries within 10 km of gas pipelines. | Discoveries+Pipelines 0.94 | Discoveries+Pipelines 0.99 |
| List gas within 5 km of oil. (ambiguous: every layer has gas and oil) | Wells+Discoveries 0.30, Discoveries+Pipelines 0.29 | Wells+Discoveries 0.24, Wells+Pipelines 0.23 |

`qwen35-4b` on other questions: "Show me all dry wells." gives Wells 1.00. "Which
injection pipelines cross condensate discoveries and are within 2 km of salt wells?"
gives Pipelines+Discoveries+Wells 1.00. "Pipelines carrying gas near the TROLL field
wells" gives Pipelines+Wells 0.92.

## 9. Error analysis

Across all 11 trained runs, 56 test questions are missed by at least one run. Only 4
are missed by every run:

| Gold | Question | Why |
|---|---|---|
| Wells+Discoveries | Find oil/gas/condensate wells with formation tops that are within 50 miles of gas. | "gas" may be a pipeline or a discovery |
| Wells+Discoveries | List gas wells with completion date before 2000 that are within 15 kilometers of condensate. | same, "condensate" |
| Wells+Discoveries | List all oil/gas wells within 15 kilometers of gas. | same |
| Pipelines+Wells | Show oil wells with a current phase of 'ABANDONED IN PLACE' or 'DECOMMISSIONED' that are within 1 kilometer of salt pipelines. | **label noise** (below) |

Most errors follow the first pattern. The secondary layer is named only by a subtype
word shared by Pipelines and Discoveries, so the text does not determine the label.
The model learns the generator's base rates and gives these questions 0.5-0.65
confidence, which the gate catches.

**Label noise in `FELN.json`.** The humanize step damaged at least two test rows:

- Gold `['Pipelines', 'Wells']` with where `PipelinesType = 4` (oil) and
  `content_type = 17` (salt). Source: "Show me oil where current phase is ... The returned
  **pipelines** must be within 1 kilometer of salt." The humanized text says "oil
  **wells** ... within 1 kilometer of salt **pipelines**", which swaps the two layers.
  Every model answers Wells+Pipelines, which matches the text.
- Gold `['Wells', 'Pipelines']`. Source: "List gas with geochemical information. The
  returned **wells** must be within 5000 meters of condensate." The humanized text drops
  "The returned wells", so no layer is named at all.

`FELN.json` was not changed. Re-checking humanized rows whose layer nouns disagree with
`meta.layers` would remove this noise.

## 10. Measurement notes

- **Two accuracy numbers.** `metrics.json` holds `after.accuracy` from
  `FastDecisionModel.evaluate` and `test_breakdown.exact` from per-row
  `FastDecisionModel.predict`. They agree for 7 runs and differ by 1-3 questions for
  `gemma4-e2b`, `qwen35-2b`, `q08-r16-seed1` and both Laya runs. It is not batching:
  on reloaded `laya/merged`, `evaluate` gives 90.7% with batch size 1 and with the
  default batch size, while `predict` gives 90.3%. The cause inside Unsloth was not
  found. This report uses the `predict` numbers, because that is the serving path.
- **Laya is not deterministic.** Two runs with the same seed gave 92.0% and 90.3%.
  The LLM runs were not repeated with the same seed.
- **Data provenance.** The test questions come from the same generator and humanizer
  as the training questions. Accuracy on real user wording is unmeasured.

## 11. Problems hit

| Problem | Fix |
|---|---|
| First download of Qwen3.5-4B stalled on Xet | Unsloth retried automatically with `HF_HUB_DISABLE_XET=1` |
| Several runs downloading the same model at once could race in the HF cache | Pre-downloaded with `snapshot_download`, then staggered launches by 45 s |
| `load_in_4bit` not supported for Laya | `--no-4bit` flag in `train.py` |
| Laya crashed at save: `'DecisionModel' object has no attribute 'save_pretrained'` | `train.py` saves merged when there is no adapter save; Laya re-run as `laya`, first run kept as `laya-nosave` (metrics only) |
| `predict.py` loaded everything in 4-bit | Loads `adapter` folders in 4-bit and `merged` folders in 16-bit |

## 12. Recommendation

- **Default: `q08-r16` (Qwen3.5-0.8B, r16).** It ties the 4B, has the best calibration
  (ECE 0.018), is 1.7 GB merged and trains in under 5 minutes. `q08-r64` is the same
  size with 3 fewer misses, which is within noise.
- **Edge / CPU: `laya`.** It is 808 MB and ModernBERT-based, about 4 points behind.
- **Use the confidence.** At ≥ 0.7, 89% of questions are answered at 98.5%. Send the
  rest to a fallback (ask the user, or a larger model).
- **Better data before bigger models.** The ceiling (about 95%) comes from ambiguous
  and mislabelled questions, not model capacity. Either fix the humanizer so it keeps
  layer nouns, or accept that "within X km of gas" is ambiguous and let the gate handle
  it.

## 13. Reproduce

```bash
# Mac, in this folder
python3 prepare.py ~/Documents/ArcGIS/Projects/NorthSea/FELN.json data
rsync -a prepare.py train.py predict.py data gc1:feln-unsloth/

# gc1
ssh gc1
cd ~/feln-unsloth
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python --upgrade unsloth datasets
tmux new -s feln-unsloth            # or: tmux attach -t feln-unsloth
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --model unsloth/Qwen3.5-0.8B --out runs/q08-r16 2>&1 | tee logs/q08-r16.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/python train.py --model unsloth/gemma-4-E4B-it --out runs/gemma4-e4b
CUDA_VISIBLE_DEVICES=2 .venv/bin/python train.py --model convaiinnovations/laya --no-4bit --out runs/laya
# other flags: --rank 64, --epochs 5, --seed 1, --max-steps 20 (smoke)

.venv/bin/python predict.py runs/q08-r16/adapter "Find oil discoveries within 10 km of gas pipelines."
```

Serve a merged model through the Decision API (from the Unsloth guide, not tested here):

```bash
UNSLOTH_SYSTEMONE_MODEL=$HOME/feln-unsloth/runs/q08-r16/merged unsloth studio -H 0.0.0.0 -p 8888
```

## 14. Artifacts

On gc1 under `~/feln-unsloth/runs/<run>/`:

- `adapter/`: LoRA weights plus the decision head and calibrated temperatures,
  113-330 MB. Not present for Laya.
- `merged/`: 16-bit full model for the Decision API. Not present for `smoke`,
  `laya-nosave`.
- `metrics.json`: before, after, calibration on val, test breakdown.
- `test_predictions.jsonl`: one line per test question with gold, prediction and all
  12 probabilities.

In this repository:

| Path | What |
|---|---|
| `prepare.py` | `FELN.json` → decision rows, stratified split, baselines; defines `QUESTIONS` |
| `train.py` | fine-tune, calibrate on val, score test, save (flags: `--model --rank --epochs --seed --max-steps --no-4bit`) |
| `predict.py` | CLI: questions → answer and top-3 probabilities |
| `app.py`, `index.html` | web tester (section 15) |
| `data/{train,val,test}.jsonl` | the exact rows every run used |
| `runs/<run>/{metrics.json,test_predictions.jsonl}` | results of every run (weights stay on gc1) |
| `logs/*.log` | training logs copied from gc1 |
| `tasks/todo.md` | plan and running review notes |

## 15. Web tester

`app.py` serves `index.html` and two endpoints on gc1 (`127.0.0.1:8095`, ran in tmux
window `feln-unsloth:web`; stopped 2026-10-09, along with the SSH tunnel): `GET /api/models` lists every run with an `adapter/` (plus
`laya/merged`), and `POST /api/decide` runs `FastDecisionModel.predict` with the same
`QUESTIONS` used in training. Models load lazily and stay cached, and a lock keeps GPU
calls one at a time. Measured: first request 14 s (load), then about 85 ms per
question for `q08-r16`. Unsloth Studio's own Decision API (`unsloth studio`) was not
used: it is a separate app with its own setup, and its FastAPI dependencies were not
installed in this venv.

## 16. References

- Unsloth, *Train your own Decision Model with Unsloth*:
  <https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth>
- Unsloth, decision models / Decision API (Laya, Jev): <https://unsloth.ai/docs/models/decision-laya>
- Laya checkpoint: <https://huggingface.co/convaiinnovations/laya> (code: `~/GWorkspace/laya`)
- Dataset format reference: <https://huggingface.co/datasets/LocalLLaMA/typed-decisions>
- FELN format and compare semantics: `~/GWorkspace/feln` (`CLAUDE.md`, `README.md`)
- Catalog: `~/Documents/ArcGIS/Projects/NorthSea/Layers.json`
