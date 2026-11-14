"""Saved facts by meaning (#204): the cases :data:`app.ai.facts.MEANING_LIFT`
and :data:`app.ai.facts.MEANING_GAP` are set from.

A household's saved facts (:data:`FACTS`), then app asks that are writes (a
read never takes a fact by meaning), none sharing two match terms with any
fact, so ``facts.related`` sends nothing and every fact is meaning's alone:

- :data:`PARAPHRASES` mean one fact in other words: it should go with them.
- :data:`CONTROLS` mean none: nothing may go ("never guessed"). Some are near
  misses on purpose ("resign the game" beside the chess fact).

``test_fact_cases.py`` holds the data to that in CI and, with
``CONDUCTOR_EMBEDDINGS_LIVE=1``, runs it through ``fact_meaning.nearest``
against the embeddings service on 8600, printing each case's lift and gap.

Measured 2026-11-13 (EmbeddingGemma 2 Q8_0 on llama.cpp; lift over the
median cosine with ``facts.BACKGROUND``): the meant fact is the nearest for
14/14 paraphrases, lift +0.06 to +0.16. Controls reach +0.10 ("move my haircut
to Monday", gap 0.009) and "resign the game" +0.08 with a 0.06 gap beside the
chess fact. The bar (lift >= 0.11 and gap >= 0.05) sends 6/14 paraphrases'
facts, none wrong, and 0/12 controls'. Reads are left out because topic is not
relevance: "show my last chess game" lifts +0.19 on the chess fact and "what's
on my calendar" +0.17 on a birthday, past every paraphrase.
"""

from __future__ import annotations

FACTS: tuple[str, ...] = (
    "I sort music on Sundays",
    "focus time is 9 to 11 on weekdays",
    "the spare key is under the blue pot",
    "the recycling goes out on Thursdays",
    "I'm allergic to shellfish",
    "my sister's birthday is March 14",
    "the dog eats twice a day, at 7 and 6",
    "I only play chess with black against the computer",
    "we don't play music after 10pm, the baby sleeps",
    "the car takes synthetic oil",
    "I prefer vinyl rips over streaming quality",
    "the plumber we trust is Dave from Ace",
)

# (the ask's request, the fact it means)
PARAPHRASES: tuple[tuple[str, int], ...] = (
    ("add a task to organize my albums", 0),
    ("schedule deep work tomorrow", 1),
    ("block my mornings for heads-down work", 1),
    ("add a task to copy the house key for the sitter", 2),
    ("add a reminder to take the bins to the curb", 3),
    ("add a task to buy a gift for my sibling", 5),
    ("remind me to feed the puppy", 6),
    ("start a new game", 7),
    ("queue something loud for tonight", 8),
    ("add a task for an oil change", 9),
    ("add a task: book the car service", 9),
    ("download the new album in the best quality", 10),
    ("add a task to get the faucet fixed", 11),
    ("plan a seafood dinner for Friday", 4),
)

CONTROLS: tuple[str, ...] = (
    "add milk to the shopping list",
    "add a task to renew my passport",
    "mark the laundry task done",
    "queue the next Radiohead album",
    "play Abbey Road",
    "add a task to call mom",
    "resign the game",
    "add a task to paint the fence",
    "delete the dentist task",
    "make a playlist for the road trip",
    "add a task to clean the gutters",
    "move my haircut to Monday",
)
