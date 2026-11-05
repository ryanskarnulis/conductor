"""S0: 12B <-> 26B swap time through llama-swap under today's RAM load.

Each call is a tiny prompt (max 4 tokens), so its wall time is unload + load +
a negligible generation. Run: python s0_swap.py [rounds]
"""

from __future__ import annotations

import subprocess
import sys
import time

from common import OUT, append, chat

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PATH = OUT / "s0_swap.jsonl"


def mem() -> str:
    return subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1]


def ping(model: str) -> float:
    reply, _ = chat(model, [{"role": "user", "content": "Say OK."}], max_tokens=4)
    return reply.seconds


def main() -> None:
    sequence = ["gemma-4-12b", "gemma-4-12b"]
    for _ in range(ROUNDS):
        sequence += ["gemma-4-26b-a4b", "gemma-4-26b-a4b", "gemma-4-12b", "gemma-4-12b"]
    previous = None
    for model in sequence:
        seconds = ping(model)
        kind = "warm" if model == previous else "swap"
        row = {"model": model, "kind": kind, "seconds": round(seconds, 2), "mem": mem(),
               "at": time.strftime("%H:%M:%S")}
        append(PATH, row)
        print(row, flush=True)
        previous = model


if __name__ == "__main__":
    main()
