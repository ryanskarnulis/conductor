"""Recall by meaning (#147): the paraphrase cases and unrelated controls the
relevance bar (:data:`app.ai.memory.LIFT_ALONE`) is set from.

A household's worth of turns (:data:`TURNS`), then two kinds of query, none
sharing a memory word (:func:`app.ai.memory.terms`) with any turn, so word
overlap finds nothing and every hit is meaning's alone:

- :data:`PARAPHRASES` say a turn's thing in other words, terse like a
  planner's recall query or whole like the person's: recall should cite it.
- :data:`CONTROLS` are about things never said: recall must find nothing
  ("nothing found is said, never guessed"). Some are near misses on purpose
  ("passport renewal" beside "renew the car registration").

``test_recall_cases.py`` holds the data to that in CI (no service) and, with
``CONDUCTOR_EMBEDDINGS_LIVE=1``, runs it through the real recall against the
embeddings service on 8600, printing each query's lift over the median cosine.

Measured 2026-11-11 (EmbeddingGemma 2 Q8_0 on llama.cpp b11541; 36 turns):
the meant turn is the nearest for 28 of 30 paraphrases, its lift from +0.07
to +0.26 (median +0.17). Unrelated controls stay at or under +0.11, except
the near miss "passport renewal" at +0.16 (beside "renew the car
registration", a turn it shouldn't cite). A bar of +0.18 cites 11/30
paraphrases on meaning alone and 0/16 controls; +0.17 would cite 17/30 with a
margin of 0.008 over that near miss, too thin to say "never guessed". Word
overlap still finds whatever shares a word, as before.
"""

from __future__ import annotations

# The turns, oldest first: the person's words as indexed (titles inline).
TURNS: tuple[str, ...] = (
    "for the garage shelves, let's go with the steel rack",
    "grab Odessa by Caribou",
    "move my dentist appointment to Thursday afternoon",
    "add milk, eggs and bread to the shopping list",
    "remind me to call mom on Sunday",
    "download the new Radiohead album",
    "start a new chess game against the computer",
    "the fence needs two coats of paint",
    "book a table for dinner at the Italian place on Friday",
    "renew the car registration before the end of the month",
    "put the kids' soccer practice on the calendar for Tuesdays",
    "I want to repaint the bedroom a light blue",
    "file the tax documents in the finance folder",
    "make a playlist for the road trip",
    "the water heater is making a weird noise, add a task to get it checked",
    "let's plan the vegetable garden: tomatoes and peppers this year",
    "order a new filter for the furnace",
    "schedule the vet for the dog's shots",
    "sort my music into folders by genre",
    "the spare key is under the blue pot by the porch",
    "cancel the gym membership at the end of the month",
    "the laptop battery drains fast, look into a replacement",
    "find a recipe for lasagna for Sunday dinner",
    "set up a reading list: Dune, Foundation, Hyperion",
    "the sink in the downstairs bathroom is dripping",
    "pay the internet bill on the 15th",
    "queue the Beatles' Abbey Road",
    "resign the chess game, I'm lost",
    "we're hosting Thanksgiving this year, invite the Hendersons",
    "the lawn mower won't start",
    "add a task to clean the gutters before winter",
    "move the Kitchen project to next quarter",
    "grandma's birthday is March 3rd, get her a card",
    "back up the photos from the phone",
    "the smoke alarm in the hallway keeps chirping",
    "pick up the dry cleaning on Saturday",
)

# (query, the turn it means): recall should cite that turn.
PARAPHRASES: tuple[tuple[str, int], ...] = (
    ("carport storage material", 0),
    ("what did we choose for storing tools in the carport", 0),
    ("tooth doctor visit", 2),
    ("groceries we needed", 3),
    ("ring my mother", 4),
    ("boiler rattling", 14),
    ("vehicle paperwork", 9),
    ("the puppy's vaccinations", 17),
    ("football training for the children", 10),
    ("heating system part to buy", 16),
    ("what's growing in the backyard beds", 15),
    ("sleeping room wall colour", 11),
    ("songs for the long drive", 13),
    ("workout club subscription", 20),
    ("notebook PC power problem", 21),
    ("pasta bake for the weekend meal", 22),
    ("sci-fi novels to read", 23),
    ("restaurant reservation", 8),
    ("playing a match versus the bot", 6),
    ("leaky faucet", 24),
    ("broadband payment", 25),
    ("the grass cutter is broken", 29),
    ("clearing the roof channels", 30),
    ("holiday feast guests", 28),
    ("present for my grandmother", 32),
    ("the detector that beeps in the corridor", 34),
    ("laundry pickup", 35),
    ("copying pictures off my mobile", 33),
    ("where we hide the extra way in", 19),
    ("how many layers the boundary wall gets", 7),
)

# Never said: recall must find nothing.
CONTROLS: tuple[str, ...] = (
    "boat trailer hitch",
    "piano tuner visit",
    "passport renewal",
    "wedding anniversary gift",
    "printer ink cartridges",
    "neighbour's tree branch over the driveway",
    "Japanese lessons",
    "insurance claim for the hail damage",
    "bitcoin price",
    "flight to Denver",
    "swimming pool cover",
    "knitting pattern for a scarf",
    "the hockey score",
    "solar panel quote",
    "the cat's litter box",
    "what did we decide about the treehouse",
)
