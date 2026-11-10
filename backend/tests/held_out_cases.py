"""The held-out plan cases (#142): run only at milestones, never tuned against.

New phrasings and situations across every kind of request the plan groups
cover (routing, cross-app, refs, control, memory, knowledge and declines,
self, long). They were written and their expected shapes fixed before their
first run. Nothing in the planner prompt, the validator, the manifests or
these expectations may change because of what this set reads: a miss here is
reported, and the fix is measured on the ordinary groups. A case that has
been tuned against moves to ``plan_cases.py``, and a new one takes its place.

Run by ``test_plan_evals.py`` behind ``CONDUCTOR_HELDOUT_EVALS=1``. Pure data:
no I/O.
"""

from __future__ import annotations

from tests.plan_cases import READ_MUSIC, READ_TASKS, Case, Cover, Long, M, none, one, state

_RECEIPTS_WAITING = (
    'tasks asked "Trash the Old receipts project and its 3 tasks?" '
    "(pending plan p9, step s1, ref pending:p9)"
)
_PENDING_P9 = frozenset({"pending", "pending:p9", "p9"})

# The kinds, in the order a run reports them.
KINDS: tuple[str, ...] = (
    "routing",
    "cross",
    "ref",
    "control",
    "memory",
    "knowledge",
    "self",
    "long",
)

HELD_OUT: tuple[Case, ...] = (
    # --- routing ----------------------------------------------------------------------
    Case(
        "h-route-recurring",
        "make the laundry task repeat every Monday",
        (one(M("ask", "tasks", ("laundry",), destructive=False)),),
        "routing",
    ),
    Case(
        "h-route-genre",
        "file Bonobo under electronic",
        (one(M("ask", "music", ("bonobo",))),),
        "routing",
    ),
    Case("h-route-rematch", "set up a chess rematch", (one(M("open", "chess")),), "routing"),
    Case(
        "h-route-drop-project",
        "get rid of the old receipts project entirely",
        (one(M("ask", "tasks", ("receipts",), destructive=True)),),
        "routing",
    ),
    # --- cross-app ------------------------------------------------------------------------
    Case(
        "h-cross-link-slideshow",
        "download the song at https://example.org/v/77 and make a task to put it in the "
        "wedding slideshow",
        (one(M("ask", "music", ("example.org",)), M("ask", "tasks", ("slideshow",))),),
        "cross",
    ),
    Case(
        "h-cross-move-overdue",
        "is it my move in chess, and is anything overdue?",
        (one(M(("look", "open"), "chess"), READ_TASKS, ordered=False),),
        "cross",
    ),
    Case(
        "h-cross-sort-backup",
        "sort the new songs into folders, then add a task to back up the library Saturday",
        (one(M("ask", "music", ("sort",)), M("ask", "tasks", ("saturday",))),),
        "cross",
    ),
    # --- references -----------------------------------------------------------------------
    Case(
        "h-ref-passport",
        "move that passport one to next Monday",
        (one(M("ask", "tasks", ("passport",), destructive=False)),),
        "ref",
        state(recent='pcc:task:88 task "Renew passport" (created Tue, due Nov 20)'),
    ),
    Case(
        "h-ref-which-bill",
        "mark the bill one done",
        (one(M("ask_user")), one(M("ask_user"), M("ask", "tasks", ref=True))),
        "ref",
        state(
            matches='tasks: "Pay the water bill" (pcc:task:70, due Mon); '
            '"Pay the electric bill" (pcc:task:71, due Tue)'
        ),
    ),
    Case(
        "h-ref-album-playlist",
        "when that album finishes, add a task for Sunday to make a playlist out of it",
        (
            one(M("ask", "tasks", ("currents",))),
            one(READ_MUSIC, M("ask", "tasks", ref_or_words=("currents",))),
        ),
        "ref",
        state(recent='music:album:61 album "Currents" by Tame Impala (queued Tue, 9 of 13 done)'),
    ),
    # --- control ----------------------------------------------------------------------------
    Case(
        "h-control-go-ahead",
        "go ahead",
        (one(M("confirm", answer="yes")),),
        "control",
        state(waiting=_RECEIPTS_WAITING),
        _PENDING_P9,
    ),
    Case(
        "h-control-amend-pending",
        "wait, only trash the empty tasks in it and keep the project",
        (one(M("amend", target_has="p9")),),
        "control",
        state(waiting=_RECEIPTS_WAITING),
        _PENDING_P9,
    ),
    Case(
        "h-control-drop-album",
        "never mind that album download",
        (one(M("cancel", target_has="23")),),
        "control",
        state(jobs='music:job:23 downloading album "Currents" (running, 4 of 13)'),
        frozenset({"music:job:23", "job:23", "23"}),
    ),
    Case(
        "h-control-scratch-haircut",
        "scratch that, I already booked it",
        (one(M("undo")), one(M("ask", "tasks", ("haircut",), destructive=True))),
        "control",
        state(
            last='s1 ask tasks "add a task to book a haircut Friday" -> created task '
            '"Book a haircut" (pcc:task:410, due Fri; undo: trash)'
        ),
        frozenset({"pcc:task:410", "last"}),
    ),
    # --- memory -------------------------------------------------------------------------------
    Case(
        "h-memory-restaurant",
        "what was the name of that restaurant we talked about last month?",
        (one(M("recall", words=("restaurant",))),),
        "memory",
    ),
    Case(
        "h-memory-keep-in-mind",
        "keep in mind that my sister's birthday is March 3rd",
        (one(M("ask", "memory", ("sister",))),),
        "memory",
    ),
    Case(
        "h-memory-forget-flight",
        "you can forget about Dad's flight now",
        (one(M("ask", "memory", ("forget", "flight"))),),
        "memory",
        state(facts="Dad's flight lands Thursday at 6"),
    ),
    # --- knowledge and declines -----------------------------------------------------------
    Case("h-know-al-dente", "what does al dente mean?", (none("knowledge"),), "knowledge"),
    Case("h-decline-thermostat", "set the thermostat to 68", (none("decline"),), "knowledge"),
    Case(
        "h-decline-bitcoin", "what's bitcoin trading at right now?", (none("decline"),), "knowledge"
    ),
    # --- self and chat ------------------------------------------------------------------------
    Case("h-self-apps", "which apps are you connected to?", (none("self"),), "self"),
    Case("h-chat-morning", "good morning!", (none("chat"),), "self"),
    # --- long ---------------------------------------------------------------------------------
    Case(
        "h-long-saturday",
        "for Saturday: add tasks to mow the lawn, clean the gutters and wash the car, download "
        "In Rainbows by Radiohead for the drive, and tell me whose move it is in chess",
        (
            Long(
                (
                    Cover("tasks", ("ask",), ("lawn", "gutters", "car")),
                    Cover("music", ("ask",), ("rainbows",)),
                    Cover("chess", ("look", "open")),
                ),
                min_steps=3,
            ),
        ),
        "long",
    ),
    Case(
        "h-long-catch-up",
        "catch me up: anything overdue, any downloads still running, how many songs are left to "
        "sort, and what did we decide about the gutter quote?",
        (
            Long(
                (
                    Cover("tasks", ("look", "ask")),
                    Cover("music", ("look", "ask")),
                    Cover(None, ("recall",), ("gutter",)),
                ),
                min_steps=3,
            ),
        ),
        "long",
    ),
)
