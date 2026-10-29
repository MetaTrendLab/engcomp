from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Encoded:
    payload: Any
    n_tokens: int
    n_options: int
    aux: Any = None


class Backend(Protocol):
    name: str
    kind: str
    max_input_tokens: int
    max_batch_tokens: int

    def encode(self, state, question: dict) -> Encoded:
        pass

    def run(self, batch: list[Encoded]) -> list[list[float]]:
        pass

    def options(self, question: dict) -> list[str]:
        pass


def sidecar_answer(question: dict, keys: list[str], probs: list[float]) -> dict:
    p = {k: float(v) for k, v in zip(keys, probs)}
    answer = {"type": question["type"], "probabilities": p, "confidence": max(p.values())}
    if question["type"] == "noul":
        answer["noul"] = p["true"]
    elif question["type"] == "choice":
        answer["choice"] = max(p, key=p.get)
    else:
        answer["score"] = sum(int(k) * v for k, v in p.items())
    return answer
