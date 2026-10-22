# Command classification API

The API classifies one command per request. It serves the character TF-IDF and
linear SVM model, selected for lightweight, low-latency
CPU inference on the Virtech private cloud (see the [deployment guide](deployment.md)).
The service never executes submitted commands or trains a model during startup.

## Run locally

Install the frozen environment and recover the saved baseline from DVC:

```sh
uv sync --frozen --group data --group dev
uv run --frozen --group data dvc pull models/tfidf_char_linear_svm
uv run --frozen uvicorn taed2_safeagent.api.app:app --host 127.0.0.1 --port 8000
```

`SAFEAGENT_MODEL_DIR` can select another compatible saved SVM directory. The
default is `models/vm_linear_svm` when that directory exists, otherwise
`models/tfidf_char_linear_svm`. The Virtech startup script explicitly selects
`models/vm_linear_svm` so a missing deployment model produces an error. The model loads
once per server process through FastAPI lifespan. A change needs a server restart.
Another model family can implement the predictor interface in `api/model.py`
without changing the request or response format. Encoder serving is not yet
implemented by the default loader.

If loading fails, the process remains alive while readiness and prediction return
503. Fix the model configuration or recover the files, then restart the service.
Startup does not download missing weights. An explicitly selected model directory
does not fall back to another model.

## HTTP contract

| Method and path | Result |
| --- | --- |
| `GET /` | Service name |
| `GET /health` | 200 when the process responds |
| `GET /ready` | Loaded model identity, or 503 when unavailable |
| `POST /predict` | One predicted label and model identity |
| `GET /docs` and `GET /redoc` | Interactive API documentation |
| `GET /openapi.json` | Generated OpenAPI schema with success and error responses |

Swagger includes five request examples: `git status`, branch creation, branch
deletion, a destructive command submitted as text, and an empty command that
demonstrates HTTP 422 validation.
Readiness documents its 503 response, and prediction documents 422, 503 and 500.

## Short live demo

Open [the deployed Swagger page](http://nattech.fib.upc.edu:40490/docs).
Call `GET /ready` to show the loaded model and its version. Expand `POST /predict`,
select **Try it out**, and use the example selector. The deployed model returns
ALLOW for `git status` and `git branch feature`, ASK for `git branch -d feature`,
DENY for `rm -rf /`, and HTTP 422 for an empty command. Submitted commands are
only classified.

The same checks can be run from the repository without loading a local model:

```sh
python scripts/smoke_test.py --demo --url http://nattech.fib.upc.edu:40490
```

The short demo checks readiness, health, Swagger, OpenAPI, both example
predictions and invalid input. Predictions must identify the same model as
readiness. Add `--expect-model-version sha256:EXPECTED_HASH` to require the
identity obtained independently from the selected model file. A failed check
returns a nonzero exit code. These examples check service behaviour rather than
measuring model accuracy.

## Request and response

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

## Tests

```sh
uv run --frozen --group data --group dev pytest tests/test_api.py tests/test_api_model.py -q
```

Fast tests inject a controlled predictor before startup and require no saved
weights, DagsHub credentials or network downloads. They cover the three labels,
schema errors, exact input boundaries, model failures, startup loading and OpenAPI.
Loader tests use a tiny local fixture to check artifact identity and compatibility.
Real saved-model integration is a separate check, not established by these tests.
