# JobCard

JobCard is a step that moved to the background: the app keeps working after the turn ends.

## Use
While a job runs. Its strand stays open at 80% and hums; its anchor shows a progress ring. It survives the next request.

## The consumer provides
The app, the job's title, progress as `n of m`, and how to stop it in words.

## Rules
- Progress shows as numbers too, never only the bar.
- When it ends, the JobCard goes and a FinishedNote arrives.
