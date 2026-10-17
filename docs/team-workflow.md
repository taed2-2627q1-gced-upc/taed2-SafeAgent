# Team workflow

Trello records planned work, GitHub records implementation and review, and Slack
supports quick coordination. The [feature ownership guide](feature-ownership.md)
defines maintenance and review responsibilities.

## Trello and code

The project uses the [TAED2 SafeAgent board](https://trello.com/b/24wUyWhX/taed2-safeagent).
Its columns are BACKLOG, TO DO, IN PROGRESS, REVIEW and DONE.
Cards use the relevant M1, M2, M3 or M4 milestone label. Tasks that span the
project, such as coordination, general cleanup and documentation maintenance,
use General. Additional topic labels can complement the milestone label.
The [board snapshot from 2026-10-17](assets/trello-board-2026-10-17.json) exports
the open cards, their columns, assigned members and linked pull requests.
It is a dated planning snapshot. Card position alone does not prove that code
is implemented or that a check passed.

1. Select a card with a clear scope and completion criteria, and move it to
   IN PROGRESS when work starts.
2. Use its real card number in the branch name, for example
   `docs/trello-35-team-workflow`. A GitHub issue can be linked when useful,
   without creating a second task list for every card.
3. Include the card number and URL in the pull request description, then add
   the pull request URL to the card. The repository PR template prompts for it.
4. Move the card to REVIEW when the change is ready. Request the feature review
   and check the Project checks workflow before merging.
5. Move it to DONE only when its completion criteria are met. A merged PR can
   satisfy the code portion while a deployment or external setup remains open.

These existing changes were linked in both directions on 2026-10-17:

| Trello card | Implemented change | GitHub evidence |
| --- | --- | --- |
| [#17](https://trello.com/c/GC7YacZR) | First reproducible baseline | [PR 3](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/pull/3) |
| [#16](https://trello.com/c/SlxCVZdv) | Shared MLflow tracking | [PR 4](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/pull/4) |
| [#15](https://trello.com/c/TZE6FlcG) | DVC data and model pipeline | [PR 5](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/pull/5) |
| [#12](https://trello.com/c/nCFWh5pO) | Model registry and dataset links | [PR 12](https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent/pull/12) |

The links were added after those PRs merged. Their original branch names, merge
dates and recorded checks remain unchanged. The branch naming convention applies
to future work. [Card #35](https://trello.com/c/FSwRrpFu) tracks this workflow setup.

## Slack and GitHub notifications

The workspace is TAED2 - SafeAgent. Channels separate general coordination
(`#general`), code and review (`#development`), models (`#machine-learning`)
and report coordination (`#reports`).

The existing [development channel](https://app.slack.com/client/T0C1F14EM5H/C0C1GL57VB7)
already contains team updates and a link to PR 15. The
[GitHub app joined the channel](https://taed2-safeagent.slack.com/archives/C0C1GL57VB7/p1792228242866699)
on 2026-10-17, and Joel confirmed running the subscription commands.
Delivery of a real PR and CI event remains to be verified.

To configure or restore the integration, a workspace member adds the official
[GitHub app for Slack](https://slack.github.com/), grants access to this repository
and runs these commands in `#development`:

```text
/invite @github
/github subscribe taed2-2627q1-gced-upc/taed2-SafeAgent pulls reviews workflows
```

Accept the additional GitHub Actions permission if prompted. The `workflows`
subscription defaults to PR workflows targeting the default branch, matching
the project's Project checks workflow. See the
[GitHub notification guide](https://docs.github.com/en/integrations/how-tos/slack/customize-notifications).

Confirm the subscription with `/github subscribe list features`, then verify
that a real PR event and its CI result appear in the channel. Add their message
links here and update card #35 after verification. Account access alone does
not establish that automatic notifications are active.

## Recording decisions

Slack threads support questions, blockers and quick coordination. Once a
technical choice is accepted, summarize the decision, reason and affected card
in its GitHub issue or PR, linking the Slack thread when useful. Durable behavior
and reproduction steps also belong in the relevant project documentation.
The Trello card links that record so the decision remains visible after the
conversation moves on. A chat agreement does not replace GitHub review or CI.
