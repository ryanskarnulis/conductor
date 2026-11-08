# TimelineItem

TimelineItem is one ledger line: a timestamp and the event the screen was drawn from.

## Use
At the bottom of the panel, in a `details` element, the last eight lines, the newest in `ink` (`aria-current`). Closed on phone.

## The consumer provides
The ledger events as `time · event`, from the run record.

## Rules
- The panel and the scene are pure functions of these events. If it is not in the ledger, it is not on screen.
