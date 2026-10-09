"""Fine-tune a decision model on data/{train,val,test}.jsonl (see prepare.py).

    python train.py --out runs/qwen35-4b                  # full run
    python train.py --out runs/smoke --max-steps 20       # smoke test
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported  # isort: skip
from transformers import TrainingArguments


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def breakdown(rows: list[dict], preds: list[str]) -> dict:
    """Exact match, plus the parts: primary layer, layer set, layer count."""
    gold = [r["gold"]["layers"] for r in rows]
    parts = lambda s: s.split("+")  # noqa: E731
    n = len(gold)
    by_count = Counter(len(parts(g)) for g in gold)
    hit_count = Counter(len(parts(g)) for g, p in zip(gold, preds) if g == p)
    return {
        "exact": sum(g == p for g, p in zip(gold, preds)) / n,
        "primary": sum(parts(g)[0] == parts(p)[0] for g, p in zip(gold, preds)) / n,
        "layer_set": sum(set(parts(g)) == set(parts(p)) for g, p in zip(gold, preds)) / n,
        "layer_count": sum(len(parts(g)) == len(parts(p)) for g, p in zip(gold, preds)) / n,
        "exact_by_count": {k: hit_count[k] / v for k, v in sorted(by_count.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no-4bit", action="store_true", help="16-bit base (required for Laya)")
    a = ap.parse_args()
    out, data = Path(a.out), Path(a.data)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=a.model, max_seq_length=1024, load_in_4bit=not a.no_4bit
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r=a.rank,
        lora_alpha=a.rank,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=a.seed,
    )

    rows = {s: load(data / f"{s}.jsonl") for s in ("train", "val", "test")}
    items = {}
    for s, r in rows.items():
        items[s], report = FastDecisionModel.build_dataset(r, tokenizer, model)
        assert report["skipped"] == 0, (s, report)

    metrics = {"before": FastDecisionModel.evaluate(model, tokenizer, items["test"])}
    print("test before training:", metrics["before"])

    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=items["train"],
        eval_dataset=items["val"],
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=a.epochs,
            max_steps=a.max_steps,
            learning_rate=a.lr,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            eval_strategy="epoch",
            logging_steps=10,
            output_dir=str(out / "checkpoints"),
            save_strategy="no",
            report_to="none",
            seed=a.seed,
        ),
    )
    trainer.train()

    # Temperatures fitted on val so test stays untouched.
    metrics["calibration_val"] = FastDecisionModel.calibrate(model, tokenizer, items["val"])
    metrics["after"] = FastDecisionModel.evaluate(model, tokenizer, items["test"])
    print("test after training:", metrics["after"])

    FastDecisionModel.for_inference(model)
    preds = []
    with open(out / "test_predictions.jsonl", "w") as f:
        for r in rows["test"]:
            ans = FastDecisionModel.predict(model, tokenizer, r["state"], r["questions"])["layers"]
            preds.append(ans["answer"])
            f.write(json.dumps({"text": r["state"], "gold": r["gold"]["layers"], "pred": ans["answer"],
                                "probabilities": ans["probabilities"]}) + "\n")
    metrics["test_breakdown"] = breakdown(rows["test"], preds)
    print("test breakdown:", metrics["test_breakdown"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # Laya (ModernBERT encoder) has no adapter save, only a merged one.
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out / "adapter"))
    if a.max_steps < 0 or not hasattr(model, "save_pretrained"):
        model.save_pretrained_merged(str(out / "merged"))


if __name__ == "__main__":
    main()
