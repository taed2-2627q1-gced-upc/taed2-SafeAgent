# API deployment preparation

The API can be prepared and tested before a VM is available. The current model
is the saved CPU SVM, while the final model and Virtech setup remain pending.
The service template below targets a Linux VM with systemd. It has not been
installed or tested on a project VM.

## Architecture

An HTTP request reaches Uvicorn and FastAPI, which validate one command and its
optional context. The loaded vectorizer and SVM produce one label, and the API
returns it with the model identity. Submitted commands are never executed.

| Component | Location and role |
| --- | --- |
| Client | Another process or machine sending HTTP requests |
| Uvicorn and FastAPI | One service process on the future VM, initially at `127.0.0.1:8000` |
| Saved pipeline | Local model directory on that VM, loaded once per process |
| systemd | User service manager for starting, stopping and restarting the process |
| GitHub | Source code, dependency lock and deployment template |
| DagsHub DVC storage | Model recovery before the service starts |

Model recovery is a separate setup step. Serving does not require a DagsHub
connection for each prediction, and startup does not train or download weights.
One worker keeps one model instance in memory. The initial loopback binding
allows local checks and SSH forwarding before network access is configured.

## Prepare the runtime

Use a reviewed checkout containing the API and deployment files. The template
expects the repository at `$HOME/taed2-SafeAgent` on the VM. If its location is
different, edit `WorkingDirectory` in the installed unit. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first, then run:

```sh
cd "$HOME/taed2-SafeAgent"
uv sync --frozen --no-dev
uv run --frozen --no-dev dvc pull models/tfidf_char_linear_svm
```

Configure DVC access locally as described in [getting started](getting-started.md).
Do not put credentials in the service unit or commit them. The runtime uses the
locked Python 3.11 environment without the optional QA or encoder training groups.
Stop here if model recovery fails.

For a foreground check, start the same API used by the service:

```sh
uv run --frozen --no-dev uvicorn taed2_safeagent.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

From another terminal, run:

```sh
uv run --frozen --no-dev python -m taed2_safeagent.api.smoke --url http://127.0.0.1:8000
```

The checker prints a JSON result and exits with 0 only when health, readiness,
one prediction, empty command rejection, unknown field rejection and batch
rejection pass. It also checks that readiness and prediction identify the same
model. The prediction uses `git status` with an empty context object. Its label
is not an accuracy assertion or permission to run the command.

`--timeout` sets seconds per HTTP request, with a default of 5. Network errors,
unexpected status codes, invalid JSON and contract mismatches produce a nonzero
exit. `--expect-model-version` also rejects a different loaded artifact. On Linux,
the version of the recovered baseline can be checked independently:

```sh
MODEL_VERSION="sha256:$(sha256sum models/tfidf_char_linear_svm/pipeline.joblib | cut -d ' ' -f1)"
uv run --frozen --no-dev python -m taed2_safeagent.api.smoke --url http://127.0.0.1:8000 --expect-model-version "$MODEL_VERSION"
```

Stop the foreground process before starting the systemd service on the same port.

## Install the user service

For the first installation, run these commands on the future Linux VM from the
repository directory. Keep the configured `api.env` file when updating the unit:

```sh
install -d -m 700 "$HOME/.config/safeagent"
install -d -m 755 "$HOME/.config/systemd/user"
install -m 600 deploy/api.env.example "$HOME/.config/safeagent/api.env"
install -m 644 deploy/safeagent.service "$HOME/.config/systemd/user/safeagent.service"
systemctl --user daemon-reload
systemctl --user enable --now safeagent.service
systemctl --user status safeagent.service
```

The environment example sets `UVICORN_HOST=127.0.0.1` and `UVICORN_PORT=8000`.
The default model directory is inside the checkout. To use another compatible
SVM directory, add `SAFEAGENT_MODEL_DIR` with its absolute path to the installed
`api.env` file. Values are literal paths rather than shell expressions.
The current loader does not support encoders yet.

systemd passes the environment file to the process. Uvicorn's `--env-file` option
is not used because it configures application variables rather than its own
`UVICORN_*` settings. See [Uvicorn settings](https://uvicorn.dev/settings/).
The service runs the existing virtual environment directly, so a restart does
not install packages or retrieve models.

The unit restarts a failed process after five seconds. A missing model leaves
the process alive with readiness 503, so it needs model recovery and a manual
restart. An active systemd unit alone does not prove that inference is ready.
See the [systemd service reference](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html).

For operation after logout or at boot, confirm whether user lingering is enabled:

```sh
loginctl show-user "$USER" -p Linger
```

If it is disabled, the VM administrator can enable it with
`sudo loginctl enable-linger "$USER"`. This is a separate VM setting. See
[loginctl](https://www.freedesktop.org/software/systemd/man/latest/loginctl.html).

## Restart and diagnose

After changing the installed environment file or replacing the selected model:

```sh
systemctl --user restart safeagent.service
```

After startup finishes, check readiness and inference. If the check fails,
inspect the service logs:

```sh
uv run --frozen --no-dev python -m taed2_safeagent.api.smoke --url http://127.0.0.1:8000
journalctl --user -u safeagent.service -n 50 --no-pager
```

Changes to the installed unit also require `systemctl --user daemon-reload`
before restarting. For code or dependency updates, stop the service, select the
reviewed revision, run the frozen sync and recover its model, then start the
service and repeat the HTTP check. Keep the previous revision and model identity
so a failed update can return to that version.

Use `systemctl --user stop safeagent.service` to stop it. Automatic restart does
not override an explicit stop.

## Access from another machine

Once SSH works, a tunnel can reach the loopback service without changing its
binding. Replace `USER` and `VM_HOST` with the actual VM account and address:

```sh
ssh -N -L 8000:127.0.0.1:8000 USER@VM_HOST
```

From the client machine, run the same smoke command against
`http://127.0.0.1:8000` or open `/docs` in a browser. The client's local port 8000
must be free. If another port is forwarded, use it in the checker URL.

Direct VM access needs the actual VPN, firewall or NAT configuration. Any change
to the host binding and access policy belongs to that deployment setup. The API
currently has no authentication layer. An external URL and a successful check
from another machine remain pending until the VM is available.

## Automated checks

```sh
uv run --frozen --group data --group dev pytest tests/test_api_smoke.py -q -rs
```

The tests start Uvicorn on temporary loopback ports and exercise real HTTP. Fast
cases use a controlled predictor. The marked integration case uses the recovered
baseline, or reports an explicit skip if it is absent. No test downloads weights
or uses the reserved test partition. CI already creates the baseline through DVC
before pytest, so the saved model case runs there without remote credentials.
