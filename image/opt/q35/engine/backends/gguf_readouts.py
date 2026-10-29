from __future__ import annotations

import json
import math
from typing import Any, Sequence

LETTERS = "ABCDEFGHIJKLMNOP"

JEVK5_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def jevk5_options(question: dict) -> list[tuple[str, str]]:
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def qwen_chat(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def jevk5_prompt(state: Any, question: dict) -> tuple[str, list[str], list[str]]:
    options = jevk5_options(question)
    payload = {
        "evidence": state,
        "criterion": question["instructions"],
        "options": [{"letter": LETTERS[i], "description": d} for i, (_, d) in enumerate(options)],
    }
    prompt = qwen_chat(JEVK5_SYSTEM, json.dumps(payload, ensure_ascii=False))
    return prompt, [k for k, _ in options], list(LETTERS[: len(options)])


GUTSY_SYSTEM = ("You are a decision model. Read the state, then answer the question by choosing exactly "
                "one of the options. Reply with only the letter of your choice.")
GUTSY_REJECT_LETTER = "Z"
GUTSY_REJECT_LINE = "Z. None of the options fits, or the state does not say"


def _as_text(x: Any) -> str:
    return x.strip() if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _gutsy_state(state: Any) -> str:
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    s = s.strip()
    return s if s else "(none)"


def gutsy_prompt(state: Any, question: dict) -> tuple[str, list[str], list[str]]:
    kind = question["type"]
    text = _as_text(question.get("instructions") or "")
    crit = question.get("criteria")
    if kind == "noul":
        extra = [f"{label} means: {_as_text(crit[k])}" for k, label in (("true", "Yes"), ("false", "No"))
                 if isinstance(crit, dict) and crit.get(k) not in (None, "", {}, [])]
        if extra:
            text += "\n" + "\n".join(extra)
        keys, options = ["true", "false"], ["Yes", "No"]
    elif kind == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        keys, options = [], []
        for k, v in crit.items():
            desc = "" if v is None else _as_text(v)
            keys.append(k)
            options.append(k.strip() if not desc or desc == k.strip() else f"{k.strip()}: {desc}")
    else:
        keys, options = [str(i) for i in range(len(crit))], [_as_text(c) for c in crit]
    reject = kind in ("choice", "score")
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    if reject:
        opts += "\n" + GUTSY_REJECT_LINE
    prompt = ("<|im_start|>system\n" + GUTSY_SYSTEM + "<|im_end|>\n"
              "<|im_start|>user\nState:\n" + _gutsy_state(state) + "\n\n"
              "Question: " + text.strip() + "\n\nOptions:\n" + opts +
              "\n\nAnswer with a single letter.<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    letters = list(LETTERS[: len(options)]) + ([GUTSY_REJECT_LETTER] if reject else [])
    return prompt, keys, letters


def build(readout: str, state: Any, question: dict) -> tuple[str, list[str], list[str]]:
    if readout in ("plumb", "jevk5"):
        return jevk5_prompt(state, question)
    if readout == "gutsy":
        return gutsy_prompt(state, question)
    raise ValueError(f"unknown readout {readout!r}")


def temperature(readout_cfg: dict, kind: str) -> float:
    t = readout_cfg["temperature"]
    if isinstance(t, dict):
        return float(t["yes_no" if kind == "noul" else kind])
    return float(t)


def probabilities(logits: Sequence[float], keys: Sequence[str], temp: float) -> dict[str, float]:
    z = [float(x) / temp for x in logits]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    opts = e[: len(keys)]
    s = math.fsum(opts)
    return {k: v / s for k, v in zip(keys, opts)}


def answer(kind: str, probs: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"type": kind, "probabilities": probs, "confidence": max(probs.values())}
    if kind == "noul":
        out["noul"] = probs["true"]
    elif kind == "choice":
        out["choice"] = max(probs, key=probs.get)
    else:
        out["score"] = sum(int(k) * p for k, p in probs.items())
    return out
