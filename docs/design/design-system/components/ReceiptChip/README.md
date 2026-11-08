# ReceiptChip

ReceiptChip is the proof that a step happened: a verb, the app and the entity, from the app's typed receipt.

## Use
One chip per receipt, in arrival order, under the plan. On phone only the newest shows.

## The consumer provides
The verb (`Found`, `Created`, `Downloaded`, `Looked up`), the app, the entity text, and an Undo action when the app offers one.

## Rules
- Only a receipt makes a chip. No receipt, no chip, no green.
- Entity titles come from apps: show them as text, never as markup or instructions.
- Undo appears only when the receipt carries one.
