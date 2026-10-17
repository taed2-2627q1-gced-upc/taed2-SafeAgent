# Command classification API

The API classifies one command per request. It starts with the saved character
TF-IDF and linear SVM baseline. The final deployment model is still undecided.
The service never executes submitted commands or trains a model during startup.

## Run locally

Install the frozen environment and recover the saved baseline from DVC:

```sh
uv sync --frozen --group data --group dev
uv run --frozen --group data dvc pull models/tfidf_char_linear_svm
uv run --frozen uvicorn taed2_safeagent.api.app:app --host 127.0.0.1 --port 8000
```

`SAFEAGENT_MODEL_DIR` can select another compatible saved SVM directory. The
default is `models/tfidf_char_linear_svm` inside the repository. The model loads
once per server process through FastAPI lifespan. A change needs a server restart.
Another model family can implement the predictor interface in `api/model.py`
without changing the request or response format. Encoder serving is not yet
implemented by the default loader.

If loading fails, the process remains alive while readiness and prediction return
503. Fix the model configuration or recover the files, then restart the service.
Startup does not download missing weights or fall back to another model.

## HTTP contract

| Method and path | Result |
| --- | --- |
| `GET /` | Service name |
| `GET /health` | 200 when the process responds |
| `GET /ready` | Loaded model identity, or 503 when unavailable |
| `POST /predict` | One predicted label and model identity |
| `GET /docs` and `GET /redoc` | Interactive API documentation |

Example request:

```sh
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"command":"git status"}'
```

The response contains `label`, `model`, `model_version` and `input_mode`. Labels
are `ALLOW`, `ASK` and `DENY`. The model version is the SHA-256 of the loaded
`pipeline.joblib`, prefixed with `sha256:`. No confidence or probability is reported.
`ALLOW` is a model prediction, not permission for an agent to execute the command.

`context` is optional for the command baseline and does not enter its predictive
text. If supplied, it must be an object containing only `gitRemote`, `gitStatus`,
`agentTouchedFiles`, `lastUserPrompt` and `cwd`. Missing strings become null and
missing path lists become empty. `gitStatus` contains `untracked`, `modified`
and `staged` path lists. The shared input builder remains the source of model text.
A future predictor declaring `command-context` requires a context object.

Unknown fields, batches, incorrect types and empty or whitespace-only commands
return 422 before inference. Command characters are otherwise preserved exactly.
Commands have a limit of 16,384 characters. Canonical JSON for context, including
its default fields, has a limit of 32,768 characters. These are application input
limits, not tokenizer limits or a transport-level HTTP body size limit.

Missing models return 503. Inference failures or invalid output labels return
500 with a generic error. Model paths and exception details are not returned.
The service does not log raw commands or context.

## Service preparation

The [deployment guide](deployment.md) provides a Linux service template, startup
and restart commands, architecture and reusable HTTP checks. It can be prepared
locally before a VM and final model are available.

## Tests

```sh
uv run --frozen --group data --group dev pytest tests/test_api.py tests/test_api_model.py -q
```

Fast tests inject a controlled predictor before startup and require no saved
weights, DagsHub credentials or network downloads. They cover the three labels,
schema errors, exact input boundaries, model failures, startup loading and OpenAPI.
Loader tests use a tiny local fixture to check artifact identity and compatibility.

Recover the DVC model before running the separate integration check:

```sh
uv run --frozen --group data dvc pull models/tfidf_char_linear_svm
uv run --frozen --group data --group dev pytest tests/test_api_integration.py -q -rs
```

This check compares HTTP predictions with direct predictions from the saved
pipeline, with and without context, and verifies the model identity. It does not
train a model, execute submitted commands, use the reserved test partition or
measure accuracy. Missing artifacts produce an explicit skip, which does not
establish integration. The existing CI pipeline creates the baseline before
running pytest, so this check also runs against that pipeline output.

For a local server smoke check, start Uvicorn using the command above, then open
`http://127.0.0.1:8000/docs`. Check `/health` and `/ready`, send the example request
and try an empty command, which should return 422. These checks confirm local
serving only. A deployed endpoint still needs its own smoke check.
