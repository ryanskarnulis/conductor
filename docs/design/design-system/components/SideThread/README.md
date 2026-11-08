# SideThread

SideThread folds what a delegate app did into one line you can open: the calls Conductor made to it and how each ended. It grows from the chat view's delegate-call list.

## Use
Under a step or a reply, closed by default. One per app per turn (an app gets at most 3 calls a turn).

## The consumer provides
The app, a one-line summary, and each call's name and outcome from the activity note.

## Rules
- A compact activity note, never the app's raw transcript.
- A failed call shows the triangle icon and a `danger` border, with the reason in words.
