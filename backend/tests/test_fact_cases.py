"""The saved-facts-by-meaning cases (#204, ``tests/fact_cases.py``).

In CI: the data holds (no ask shares two match terms with any fact, so
``facts.related`` sends nothing and every fact is meaning's). Opt-in,
``CONDUCTOR_EMBEDDINGS_LIVE=1``: ``fact_meaning.nearest`` against the
embeddings service on 8600 (CPU only), printing each case's lift and gap;
no control may take a fact, and no paraphrase the wrong one.

    CONDUCTOR_EMBEDDINGS_LIVE=1 pytest tests/test_fact_cases.py -v -s
"""

from __future__ import annotations

import os
import statistics

import pytest

from app.ai import facts
from app.config import get_settings
from app.services import fact_meaning, turn_vectors
from tests.fact_cases import CONTROLS, FACTS, PARAPHRASES

LIVE = os.environ.get("CONDUCTOR_EMBEDDINGS_LIVE") == "1"


def test_no_ask_reaches_a_fact_by_its_words() -> None:
    for request in [request for request, _ in PARAPHRASES] + list(CONTROLS):
        assert facts.related(request, FACTS) == [], request
    assert all(0 <= target < len(FACTS) for _, target in PARAPHRASES)


@pytest.mark.skipif(not LIVE, reason="CONDUCTOR_EMBEDDINGS_LIVE=1: the embeddings service on 8600")
def test_the_bar_against_the_real_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "conductor_embeddings", True)
    monkeypatch.setattr(get_settings(), "conductor_facts_by_meaning", True)
    turn_vectors._reset()
    fact_meaning._reset()

    def measured(request: str) -> tuple[list[float], float]:
        found = fact_meaning._vectors(request, [*FACTS, *facts.BACKGROUND])
        assert found is not None, "the embeddings service on 8600 didn't answer"
        query, vectors = found
        cosines = [fact_meaning._cosine(query, vector) for vector in vectors]
        return cosines[: len(FACTS)], statistics.median(cosines[len(FACTS) :])

    print(f"\nbar: lift >= {facts.MEANING_LIFT} over the background, gap >= {facts.MEANING_GAP}")
    sent = 0
    wrong: list[str] = []
    for request, target in PARAPHRASES:
        cosines, baseline = measured(request)
        other = max(c for i, c in enumerate(cosines) if i != target)
        chosen = fact_meaning.nearest(request, FACTS)
        hit = chosen == FACTS[target]
        sent += hit
        if chosen is not None and not hit:
            wrong.append(request)
        print(
            f"  lift {cosines[target] - baseline:+.3f} gap {cosines[target] - other:+.3f} "
            f"{'sent' if hit else 'wrong' if chosen else 'none':5} {request}"
        )
    guessed: list[str] = []
    print("controls: the nearest fact's lift and gap")
    for request in CONTROLS:
        cosines, baseline = measured(request)
        top, second = sorted(cosines, reverse=True)[:2]
        if fact_meaning.nearest(request, FACTS) is not None:
            guessed.append(request)
        print(f"  lift {top - baseline:+.3f} gap {top - second:+.3f}  {request}")
    print(
        f"paraphrases sent {sent}/{len(PARAPHRASES)}, wrong {len(wrong)}; "
        f"controls guessed {len(guessed)}/{len(CONTROLS)}"
    )
    assert not guessed, f"a fact went with {guessed}"
    assert not wrong, f"the wrong fact went with {wrong}"
    # The bar trades reach for honesty: a third of the paraphrases is the floor.
    assert 3 * sent >= len(PARAPHRASES)
