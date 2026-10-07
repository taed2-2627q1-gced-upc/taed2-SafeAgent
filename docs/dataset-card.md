# Dataset Card: Shell Safety for SafeAgent

## Source and purpose

Source: [tomngdev/shell-safety-v1.1](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1).
Pinned [revision](https://huggingface.co/datasets/tomngdev/shell-safety-v1.1/tree/fee89770c315d525ef2ee42adee6ef9725a7621e):
`fee89770c315d525ef2ee42adee6ef9725a7621e`.
The old v2 URL resolved to v1.1 when checked. We use the explicit v1.1 name.
The source metadata declares MIT. Its README is retained with the raw snapshot.

The task is to classify proposed shell commands as ALLOW, ASK, or DENY.
The source contains synthetic examples for POSIX, PowerShell, and CMD.
Data preparation does not execute commands or train a classifier.

The counts below come from our full audit of the pinned JSONL files. File hashes
and byte counts are fixed in params.yaml and the raw snapshot manifest.

| Original split | Rows |
| --- | ---: |
| Train | 30,002 |
| Validation | 3,002 |
| Test | 1,003 |
| Total | 34,007 |

| Class | Original rows | Prepared rows |
| --- | ---: | ---: |
| ALLOW | 13,600 | 13,600 |
| ASK | 12,509 | 12,086 |
| DENY | 7,898 | 7,639 |
| Total | 34,007 | 33,325 |

## Source quality and cleaning

The six raw fields are command, session_context, label, category, shell and reason.
Mandatory fields have valid types in the pinned files. The context may omit
lastUserPrompt, assistantMessage and cwd. The remote may be null.

There are no repeated complete raw inputs, but reducing context to the allowed
fields reveals 680 redundant rows with consistent labels. We keep one stable
input and retain every source reference. We do not deduplicate commands alone.
Different contexts for the same command can represent different labels.

Two rows become indistinguishable after removing assistant text but have labels
DENY and ASK. Both propose `Initialize-Disk -Number 1 -PartitionStyle GPT`.
They are train line 20,323 and validation line 1,034. Both are quarantined,
without voting or relabeling. This may reflect a missing decisive feature as
well as an annotation problem. Raw records remain unchanged.

The interim provenance file maps each stable input ID to source split, line,
shell, category and reason. Quarantine retains complete excluded records.
Metadata is stored separately from predictive inputs.

## Allowed inputs

The safe_context_v1 profile contains gitRemote, gitStatus, agentTouchedFiles,
lastUserPrompt and cwd. Missing optional strings become null. Git status and
touched file lists represent state sets, so we sort and deduplicate them without
changing path case or contents.

assistantMessage is excluded because the source does not document whether it
was created before the decision. reason and category are annotation metadata.
Shell, source split, IDs, group IDs and labels are also excluded from the text.
Commands keep their original characters, including spaces and comments.

Prepared JSONL rows contain id, group_id, command, context and label.
The ID is SHA256 of the compact sorted key JSON array [command, context],
encoded with UTF8 and unescaped Unicode. It contains no target or metadata.

Use taed2_safeagent.data.inputs.build_input(example, mode):

| Mode | Text |
| --- | --- |
| command | Original command |
| command-context | COMMAND, original command, CONTEXT, compact canonical context JSON, separated by newlines |

Both modes use exactly the same IDs, rows, labels and partitions. Unknown modes
are rejected. Tokenization and maximum input lengths belong to model training.

## Strict split protocol

715 of 1,003 published test rows and 2,159 published validation rows have commands
also present in published train. These are not necessarily identical full inputs,
but they allow command memorization. Upstream splits remain provenance only.
There is one main project benchmark: an 80/10/10 group split with seed 42.

Grouping alone uses a command fingerprint. It compacts whitespace, applies
casefold, replaces digit sequences with `<num>` and repeatedly strips known
synthetic tail comments introduced by # or & rem. The marker list is wip, quick,
tmp, bump, rebuild, after edit, v2, collect logs, smoke, fix lint and fmt.
Original command inputs remain unchanged.

An edge connects rows with the same fingerprint, fingerprints with character
5 gram set Jaccard similarity of at least 0.90, or identical informative allowed
contexts. Short strings use the whole string as one gram. A context is informative
when it has files, a user prompt or a working directory. An empty context or a
remote by itself does not connect unrelated rows. Similarity candidates use an
exact prefix index tested against brute force. Connected components stay together.

Group IDs hash sorted input IDs joined by newlines. Groups are ordered by size,
then the SHA256 hash of 42: followed by the group ID. A greedy allocator minimizes
the increase in squared error for target total and class counts, divided by each
target. Ties use train, validation, then test. Labels stratify allocation but never
define related command groups.

| Project split | Rows | ALLOW | ASK | DENY | Groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 26,660 | 10,880 | 9,669 | 6,111 | 7,346 |
| Validation | 3,333 | 1,360 | 1,209 | 764 | 917 |
| Test | 3,332 | 1,360 | 1,208 | 764 | 919 |

There are 9,182 groups, 10,084 fingerprints and 403 similarity edges. The largest
group contains 1,375 rows. All classes and shells occur in each split.
Validation contains 6 commands with multiple labels across contexts, and test
contains 10. Known fingerprint, similarity and informative context connections
never cross partitions. Grouping does not prove absence of all semantic overlap.

## Quality checks and reproduction

Follow [getting started](getting-started.md). Git versions code, parameters,
locks, DVC pointers and small reports. DagsHub stores the DVC objects.
The pipeline stages are audit, prepare and validate.

Python checks schema, source identity, provenance, quarantine, prepared input
identity, group membership, exact assignment, balance and complete coverage.
Great Expectations checks each entire prepared partition with ten expectations.
Deepchecks checks full input label conflicts and sample mixing for all three
partition pairs. Its tabular integrity checks operate on serialized text,
without embeddings or predictions. This is not a trained text model evaluation.
Both libraries must pass their conditions, and checks do not sample rows.

Reports live under reports/data. Output serialization is UTF8 with LF newlines.
Reports omit execution timestamps and machine paths. The lockfile fixes compatible
library versions. PyNBLint has no notebooks to inspect. MLflow and CodeCarbon
records will be created for actual training in the next model feature.

## Limits and future experiment

The source README does not describe the generator, annotation procedure, field
chronology, session IDs or label review. These classes are dataset annotations.
We cannot prove independence of actual sessions or validate labels against a
supplied annotation policy.

The data has visible shortcuts. 24,721 rows have a reason value seen with one
class only. Every ALLOW row has a nonnull Git remote, while 5,158 ASK or DENY rows
have no remote. Only 90 distinct user prompts are used, and assistant messages
can describe a different action. A context gain may follow generator patterns
rather than useful understanding of user intent. These are quality diagnostics,
not proof that the allowed fields are post decision labels.

Conservative case, digit and comment grouping can join commands with different
semantics. The similarity threshold can also miss more distant paraphrases.
The split measures generalization under documented grouping rules within this
synthetic corpus. Real agent data is still needed for external validation.

For the context experiment, hold model family, splits, training budget, weighting
and selection protocol fixed. Fit vectorizers only on train, choose settings on
validation and leave test for final evaluation. Compare the same examples in both
modes, including missing context and ambiguous command slices. Comparing CodeBERT
command inputs against ModernBERT context inputs alone cannot isolate context
value. Consider a shuffled context control in that future experiment.
No context improvement is assumed in this data PR.
