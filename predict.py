"""Ask the trained decision model which layers each question needs.

    python predict.py runs/qwen35-4b/adapter "Find oil wells within 5 km of gas pipelines." ...
    python predict.py runs/laya/merged "..."     # merged folders load in 16-bit (Laya has no 4-bit)
"""

import sys
from pathlib import Path

from unsloth import FastDecisionModel  # isort: skip

from prepare import QUESTIONS

model, tokenizer = FastDecisionModel.from_pretrained(
    sys.argv[1], load_in_4bit=Path(sys.argv[1]).name == "adapter"
)
FastDecisionModel.for_inference(model)
for text in sys.argv[2:]:
    ans = FastDecisionModel.predict(model, tokenizer, text, QUESTIONS)["layers"]
    top = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    print(f"{text}\n  -> {ans['answer']}  " + "  ".join(f"{k}={p:.2f}" for k, p in top))
