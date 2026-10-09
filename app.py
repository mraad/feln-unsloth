"""Web tester for the trained FELN layer-selection decision models.

    .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8095
"""

import threading
import time
from pathlib import Path

from unsloth import FastDecisionModel  # isort: skip
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from prepare import QUESTIONS

RUNS = Path(__file__).parent / "runs"
DEFAULT = "q08-r16"
# Adapters load in 4-bit like training; Laya only has a 16-bit merged folder.
MODELS = {
    p.parent.name: p
    for p in sorted(RUNS.glob("*/adapter")) + sorted(RUNS.glob("laya/merged"))
    if p.parent.name != "smoke"
}
loaded: dict = {}
lock = threading.Lock()  # one GPU call at a time

app = FastAPI()


class Ask(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    model: str = DEFAULT


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/models")
def models():
    return {"models": list(MODELS), "default": DEFAULT}


@app.post("/api/decide")
def decide(ask: Ask):
    if ask.model not in MODELS:
        raise HTTPException(404, f"unknown model {ask.model}")
    with lock:
        if ask.model not in loaded:
            path = MODELS[ask.model]
            m, t = FastDecisionModel.from_pretrained(str(path), load_in_4bit=path.name == "adapter")
            FastDecisionModel.for_inference(m)
            loaded[ask.model] = (m, t)
        model, tokenizer = loaded[ask.model]
        start = time.perf_counter()
        ans = FastDecisionModel.predict(model, tokenizer, ask.text, QUESTIONS)["layers"]
        ms = (time.perf_counter() - start) * 1000
    probs = sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])
    return {"answer": ans["answer"], "probabilities": probs, "ms": round(ms, 1)}
