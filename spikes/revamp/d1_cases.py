"""D1 cases: the 21 routing goldens re-labelled as plans, plus the §1.1 scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

from pydantic import BaseModel

from plan_schema import Plan, step_text


@dataclass(frozen=True)
class M:
    do: str | tuple[str, ...]
    app: str | None = None
    words: tuple[str, ...] = ()
    ref: bool | None = None
    ref_or_words: tuple[str, ...] = ()
    destructive: bool | None = None
    answer: str | None = None
    target_has: str | None = None

    def matches(self, step: BaseModel) -> bool:
        kinds = (self.do,) if isinstance(self.do, str) else self.do
        if step.do not in kinds:  # type: ignore[attr-defined]
            return False
        if self.app is not None and getattr(step, "app", None) != self.app:
            return False
        text = step_text(step).lower()
        if any(word not in text for word in self.words):
            return False
        has_ref = "{s" in text
        if self.ref is not None and has_ref != self.ref:
            return False
        if self.ref_or_words and not (has_ref or any(w in text for w in self.ref_or_words)):
            return False
        if self.destructive is not None and getattr(step, "destructive", None) != self.destructive:
            return False
        if self.answer is not None and getattr(step, "answer", None) != self.answer:
            return False
        if self.target_has is not None and self.target_has not in getattr(step, "target", ""):
            return False
        return True


V2_SAY = {"capabilities": "self", "status": "self"}


@dataclass(frozen=True)
class Shape:
    steps: tuple[M, ...] = ()
    say: str = "result"
    ordered: bool = True

    def matches(self, plan) -> bool:
        if isinstance(plan, Plan):
            say, steps = plan.say, plan.steps
        else:  # v2: steps or a direct answer
            root = plan.root
            say = getattr(root, "answer", "result")
            steps = getattr(root, "steps", [])
            want = V2_SAY.get(self.say, self.say)
            if say != want or len(steps) != len(self.steps):
                return False
            return self._steps_match(steps)
        if say != self.say or len(steps) != len(self.steps):
            return False
        return self._steps_match(steps)

    def _steps_match(self, steps) -> bool:
        if self.ordered:
            return all(m.matches(s) for m, s in zip(self.steps, steps, strict=True))
        return any(
            all(m.matches(s) for m, s in zip(order, steps, strict=True))
            for order in permutations(self.steps)
        )


@dataclass(frozen=True)
class Case:
    id: str
    text: str
    expect: tuple[Shape, ...]
    group: str
    state: str = ""
    refs: frozenset[str] = field(default_factory=frozenset)


def state(**parts: str) -> str:
    base = {
        "now": "Thursday 2026-11-05, 18:40, web chat",
        "recent": "none",
        "jobs": "none",
        "waiting": "nothing",
        "last": "none",
        "matches": "none",
    }
    base.update(parts)
    return (
        f"- Now: {base['now']}\n"
        f"- Recent things (newest first): {base['recent']}\n"
        f"- Running jobs: {base['jobs']}\n"
        f"- Waiting on the person: {base['waiting']}\n"
        f"- Last plan: {base['last']}\n"
        f"- Matches for words in the message: {base['matches']}"
    )


def one(*steps: M, say: str = "result", ordered: bool = True) -> Shape:
    return Shape(steps=steps, say=say, ordered=ordered)


def none(say: str) -> Shape:
    return Shape(steps=(), say=say)


READ_TASKS = M(("look", "ask"), "tasks")
READ_MUSIC = M(("look", "ask"), "music")

_LAST_S3 = (
    's1 ask music "download the new Caribou album" -> queued album "Honey" '
    "(music:album:55, job music:job:20; a download has no undo); "
    's2 ask tasks "add a task to sort {s1} this weekend" -> created task "Sort Honey" '
    "(pcc:task:301, due Sat; undo: trash)"
)
_PENDING = (
    'tasks asked "Trash the Kitchen project and its 12 tasks?" '
    "(pending plan p7, step s1, ref pending:p7)"
)

CASES: tuple[Case, ...] = (
    # --- the routing goldens, as plans -------------------------------------------
    Case("tasks-due", "what's due today?", (one(READ_TASKS),), "golden"),
    Case("tasks-create", "add a task to water the plants tomorrow",
         (one(M("ask", "tasks", ("plants",), destructive=False)),), "golden"),
    Case("tasks-week", "what am I working on this week?", (one(READ_TASKS),), "golden"),
    Case("chess-play", "let's play chess", (one(M("open", "chess")),), "golden"),
    Case("chess-move", "play e4", (one(M("open", "chess")),), "golden"),
    Case("chess-status", "how is our chess game going?",
         (one(M("open", "chess")), one(M("look", "chess"))), "golden"),
    Case("chess-analysis", "analyze the current position on the board",
         (one(M("open", "chess")),), "golden"),
    Case("chess-reset", "reset the chess game", (one(M("open", "chess")),), "golden"),
    Case("chess-resign", "resign the game", (one(M("open", "chess")),), "golden"),
    Case("music-download-link", "download this song https://example.com/watch?v=abc",
         (one(M("ask", "music", ("example.com",))),), "golden"),
    Case("music-download-named", "grab the song Odessa by Caribou",
         (one(M("ask", "music", ("odessa",))),), "golden"),
    Case("music-save", "save this track to the music folder", (one(M("ask", "music")),), "golden"),
    Case("music-sort", "sort my music", (one(M("ask", "music")),), "golden"),
    Case("music-sort-ask", "what songs still need sorting?", (one(READ_MUSIC),), "golden"),
    Case("music-organize", "organize my library into folders", (one(M("ask", "music")),), "golden"),
    Case("refuse-lights", "turn off the living room lights", (none("decline"),), "golden"),
    Case("refuse-weather", "what's the weather tomorrow?", (none("decline"),), "golden"),
    Case("refuse-playback", "play some jazz music", (none("decline"),), "golden"),
    Case("confirm-delete-task", "delete the groceries task",
         (one(M("ask", "tasks", ("groceries",), destructive=True)),), "golden"),
    Case("confirm-wipe-project", "wipe the whole kitchen project",
         (one(M("ask", "tasks", ("kitchen",), destructive=True)),), "golden"),
    Case("local-capabilities", "what can you do?", (none("capabilities"),), "golden"),
    # --- cross-app -----------------------------------------------------------------
    Case("s3-album-task", "download the new Caribou album and add a task to sort it this weekend",
         (one(M("ask", "music", ("caribou",)), M("ask", "tasks", ref=True)),), "cross"),
    Case("cross-named-task",
         "grab the song Odessa by Caribou and add a task to put it on the party playlist tomorrow",
         (one(M("ask", "music", ("odessa",)), M("ask", "tasks", ref_or_words=("odessa",))),),
         "cross"),
    Case("cross-playlist-task",
         "download this playlist https://www.youtube.com/playlist?list=PL9 and remind me "
         "Sunday to sort it",
         (one(M("ask", "music", ("playlist",)), M("ask", "tasks", ref_or_words=("playlist",))),),
         "cross"),
    Case("cross-two-reads", "what's due today, and how are my downloads going?",
         (one(READ_TASKS, READ_MUSIC, ordered=False),), "cross"),
    Case("cross-three-reads",
         "what's due today, how's my chess game, and is the playlist done yet?",
         (one(READ_TASKS, READ_MUSIC, M("look", "chess"), ordered=False),), "cross"),
    Case("cross-task-then-chess", "add a task to study the Sicilian this weekend, then let's play",
         (one(M("ask", "tasks", ("sicilian",)), M("open", "chess")),), "cross"),
    Case("s15-same-again", "do the same for their other album",
         (one(M("ask", "music", ("caribou",)), M("ask", "tasks", ref=True)),
          one(M("ask", "music", ("caribou",)), M("ask", "tasks", ref_or_words=("sort",))),
          one(M("ask_user")),
          one(M("ask_user"), M("ask", "music"), M("ask", "tasks", ref=True))),
         "cross", state(last=_LAST_S3, recent='album "Honey" by Caribou (music:album:55)')),
    # --- references and resolution -------------------------------------------------
    Case("s6-that-song", "put that Caribou song in the house folder",
         (one(M("ask", "music", ("odessa",))),), "ref",
         state(recent='music:track:812 song "Odessa" by Caribou (saved Tue, in Unsorted)')),
    Case("ref-task-from-song", "add a task to make a party playlist with that Caribou song",
         (one(M("ask", "tasks", ("odessa",))),
          one(M("recall", words=("caribou",)), M("ask", "tasks", ref=True))), "ref",
         state(recent='music:track:812 song "Odessa" by Caribou (saved Tue, in Unsorted)')),
    Case("s7-dennis", "push the dennis thing to friday",
         (one(M("ask", "tasks", ("dentist", "friday"))),), "ref",
         state(now="Thursday 2026-11-05, 07:55, kitchen speaker (voice, no screen)",
               matches='tasks: "Dentist appointment" (pcc:task:44, due Wed)')),
    Case("ref-ambiguous", "push the dentist thing to friday",
         (one(M("ask_user")), one(M("ask_user"), M("ask", "tasks", ref=True))), "ref",
         state(matches='tasks: "Dentist appointment" (pcc:task:44, due Wed); '
               '"Call the dentist about the bill" (pcc:task:61, due Thu)')),
    # --- control: confirm, cancel, amend, undo -----------------------------------------
    Case("s9-yes", "yes", (one(M("confirm", answer="yes")),), "control",
         state(waiting=_PENDING), frozenset({"pending", "pending:p7", "p7"})),
    Case("s9-no", "no, leave it", (one(M("confirm", answer="no")),), "control",
         state(waiting=_PENDING), frozenset({"pending", "pending:p7", "p7"})),
    Case("s10-stop", "actually, stop", (one(M("cancel", target_has="31")),), "control",
         state(jobs="music:job:31 sorting the whole library (running, 112 of 400 filed)"),
         frozenset({"music:job:31", "job:31", "31"})),
    Case("amend-job", "actually just grab the first ten songs",
         (one(M("amend", target_has="17")),), "control",
         state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)'),
         frozenset({"music:job:17", "job:17", "17"})),
    Case("s11-undo", "undo that", (one(M("undo")),), "control",
         state(last=_LAST_S3), frozenset({"pcc:task:301", "music:album:55", "last"})),
    # --- memory ----------------------------------------------------------------------
    Case("s5-garage", "what did we decide about the garage shelves last week?",
         (one(M("recall", words=("garage",))),
          one(M("recall", words=("garage",)), READ_TASKS)), "memory"),
    Case("memory-caribou", "when did I last download something by Caribou?",
         (one(M("recall", words=("caribou",))), one(M("ask", "music", ("caribou",)))), "memory"),
    # --- general knowledge and declines ------------------------------------------------
    Case("knowledge-egg", "how long do I boil an egg?", (none("knowledge"),), "knowledge"),
    Case("knowledge-capital", "what's the capital of Portugal?", (none("knowledge"),), "knowledge"),
    Case("decline-news", "what's in the news today?", (none("decline"),), "knowledge"),
    # --- hard planning -------------------------------------------------------------------
    Case("s14-plan-week", "plan my week around the garage project",
         (one(M("think")), one(READ_TASKS, M("think"))), "hard"),
    # --- self-report and chat -----------------------------------------------------------
    Case("status-now", "what are you working on right now?",
         (none("status"), one(M("look", "music"))), "self",
         state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)')),
    Case("status-last", "what did you just do?", (none("status"),), "self", state(last=_LAST_S3)),
    Case("chat-thanks", "thanks!", (none("chat"),), "self"),
)


# Held out: written after v1's misses were read, never used to tune v2.
HOLDOUT: tuple[Case, ...] = (
    Case("ho-swim-task", "download Swim by Caribou and add a task to sort it Monday",
         (one(M("ask", "music", ("swim",)), M("ask", "tasks", ref_or_words=("swim",))),),
         "holdout"),
    Case("ho-lyrics-task",
         "find the song that goes 'I can't do without you' and add a task to put it on the "
         "party playlist",
         (one(M("ask", "music", ("without you",)), M("ask", "tasks", ref=True)),), "holdout"),
    Case("ho-two-reads", "how many songs are waiting to be sorted, and what's overdue?",
         (one(READ_MUSIC, READ_TASKS, ordered=False),), "holdout"),
    Case("ho-cancel", "cancel that download", (one(M("cancel", target_has="17")),), "holdout",
         state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)'),
         frozenset({"music:job:17", "job:17", "17"})),
    Case("ho-amend", "make it the first twenty instead", (one(M("amend", target_has="17")),),
         "holdout",
         state(jobs='music:job:17 downloading playlist "Road Trip" (running, 6 of 40)'),
         frozenset({"music:job:17", "job:17", "17"})),
    Case("ho-recall", "what do you remember about the fence project?",
         (one(M("recall", words=("fence",))), one(M("recall", words=("fence",)), READ_TASKS)),
         "holdout"),
    Case("ho-knowledge", "how many teaspoons are in a tablespoon?", (none("knowledge"),),
         "holdout"),
    Case("ho-decline-score", "what's the score of the Leafs game?", (none("decline"),),
         "holdout"),
    Case("ho-delete-project", "delete the fence project",
         (one(M("ask", "tasks", ("fence",), destructive=True)),), "holdout"),
    Case("ho-chess-last", "how did my last chess game go?",
         (one(M("look", "chess")), one(M("open", "chess"))), "holdout"),
)
