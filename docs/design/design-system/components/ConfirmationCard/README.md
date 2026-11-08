# ConfirmationCard

ConfirmationCard asks before a write: the app that owns the write asks its question, and the run waits on you.

## Use
One at a time, right under the plan. The waiting step's strand and anchor pulse amber, and Warden turns amber.

## The consumer provides
The app's name, its question passed through verbatim, why it asks, and two actions: the primary names the action ("Add it", never "OK"), the other is "Not now".

## Rules
- Only a question from the last turn can take a yes; any other lapses it.
- Voice and tap are equal: "yes" does what the primary button does.
- 44px buttons.
