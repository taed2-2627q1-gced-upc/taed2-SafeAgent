# Virtech deployment and runtime guide

This guide documents the setup, runtime environment, model selection, service lifecycle,
and smoke testing for the **SafeAgent API** deployed on **Virtech** (the private OpenNebula
cloud of the Facultat d'Informàtica de Barcelona, UPC).

The short presentation demo uses the public API at
`http://nattech.fib.upc.edu:40490/docs`. The [API guide](api.md#short-live-demo)
shows the manual steps and the automatic check command.

## Model selection rationale

For the Virtech deployment and live API demonstration, the team selected the
**Safety-Calibrated Linear SVM (`models/vm_linear_svm`)**, trained on the complete dataset ($N = 33,325$).

This serving artifact uses train, validation and test examples for fitting, so it
has no independent test score. Research results use the separate baseline artifact
and its original partitions. Deployment training rejects baseline output paths.

While neural transformer models (such as ModernBERT-base with execution context) achieved
stronger contextual accuracy during offline GPU experiments, their forward pass on a
single CPU without hardware acceleration incurs multi-second latency per request. Because
shell safety filtering requires immediate real-time feedback (sub-100 ms target) and the
Virtech VM is allocated a single vCPU without a GPU, the lightweight linear SVM model is
the technically sound choice:

* **CPU inference latency**: < 5 ms per request.
* **Memory footprint**: < 50 MB RAM for model weights and vocabulary.
* **Dependency overhead**: Pure scikit-learn pipeline, avoiding PyTorch/CUDA runtime overhead.

The full experimental comparison, contextual trade-offs, and environmental metrics are
analyzed in the project report.

## Infrastructure and network architecture

Virtech hosts Linux virtual machines with network virtualization managed through NAT.
When selecting a template with **sense VPN** (external NAT ports), the machine is accessible
from the internet for the application ports without needing the UPClink VPN.
The standard SSH port requires the campus network or VPN, as described in the
[Virtech student guide](https://www.fib.upc.edu/sites/fib/files/documents/fib/serveis-tic/private_cloud_service_for_students.pdf).

### Specifications
* **Platform**: OpenNebula (Virtech FIB UPC - `https://virtech.fib.upc.edu`)
* **Base OS**: Ubuntu 20.04 LTS / Ubuntu 24.04 LTS
* **Compute resources**: 1 vCPU, 4 GB RAM, 20 GB Disk
* **User**: `alumne` (with `sudo` privileges)

### NAT port mapping
Each machine receives an internal IPv4 address in the `172.16.4.XXX` range. External
connections are routed through `nattech.fib.upc.edu` (`147.83.148.217`):

| Service | Internal address | External public address |
| --- | --- | --- |
| **SSH** | `172.16.4.XXX:22` | `nattech.fib.upc.edu:220XXX` (`22000 + XXX`) |
| **HTTP API** | `172.16.4.XXX:8080` | `nattech.fib.upc.edu:40XXX0` (`40000 + (XXX*10)`) |
| **Alternative ports** | `172.16.4.XXX:8081-8089` | `nattech.fib.upc.edu:40XXX1-40XXX9` |

Inside the VM, running `/etc/update-motd.d/99-info-nat` displays the exact mapping.

## Step 1: Connecting to the VM

Connect via SSH using the mapped external port:

```bash
ssh alumne@nattech.fib.upc.edu -p 220XXX
```

Enter the password chosen during VM creation in the OpenNebula template.

## Step 2: Automated runtime provisioning

SafeAgent includes an automated setup script that installs `uv`, configures the Python 3.11
environment with frozen dependencies, configures DVC credentials locally, and downloads the
model artifact.

On the VM:

```bash
git clone https://github.com/taed2-2627q1-gced-upc/taed2-SafeAgent.git
cd taed2-SafeAgent
bash scripts/virtech_setup.sh
```

The script will:
1. Verify and install `uv` into `$HOME/.local/bin`.
2. Run `uv sync --frozen --group data` to download CPython 3.11 and create the exact locked virtual environment.
3. Configure DVC storage credentials securely (see below).
4. Run `dvc pull models/vm_linear_svm` to retrieve the trained model weights.
5. Validate that `taed2_safeagent.api.model.load_predictor()` initializes cleanly.

## Step 3: Secure DVC credential management

Artifacts are stored on DagsHub S3 storage (`https://dagshub.com/Pau-Balaguer/taed2-SafeAgent.s3`).
In accordance with project rules, **credentials must never be committed to Git**.

On the VM, credentials can be provided in two secure ways:

### Option A: Local DVC configuration (Recommended)
Store the token in `.dvc/config.local`, which is ignored by `.dvc/.gitignore`:

```bash
uv run --frozen --group data dvc remote modify --local storage access_key_id <DAGSHUB_TOKEN>
uv run --frozen --group data dvc remote modify --local storage secret_access_key <DAGSHUB_TOKEN>
```

### Option B: Environment variables
Supply AWS S3-compatible environment variables in the shell session:

```bash
export AWS_ACCESS_KEY_ID="<DAGSHUB_TOKEN>"
export AWS_SECRET_ACCESS_KEY="<DAGSHUB_TOKEN>"
```

Then run `uv run --frozen --group data dvc pull models/vm_linear_svm`.

## Step 4: Service lifecycle management (start and restart)

SafeAgent provides `scripts/api_service.sh` for process lifecycle management on the VM,
tracking the PID in `api.pid` and streaming server output to `api.log`.

### Start the API service
```bash
bash scripts/api_service.sh start
```
Starts Uvicorn in the background bound to `0.0.0.0:8080`, validates startup, and logs output.

### Check service status
```bash
bash scripts/api_service.sh status
```
Reports the active PID, port, and checks the local `/health` endpoint.

### Restart the service
```bash
bash scripts/api_service.sh restart
```
Gracefully terminates the existing server process and starts a fresh instance.

### Stop the service
```bash
bash scripts/api_service.sh stop
```

## Step 5: Smoke testing and client verification

A dedicated smoke test client (`scripts/smoke_test.py`) verifies that the deployed API
correctly implements the HTTP contract for both **valid** and **invalid** requests.

From an external client machine (e.g. developer laptop):

```bash
python scripts/smoke_test.py --url http://nattech.fib.upc.edu:40XXX0
```

### Verification scenarios covered

The suite executes 11 distinct checks:
1. **Service root (`GET /`)**: Validates HTTP 200 and `{"service": "SafeAgent"}` response.
2. **Liveness check (`GET /health`)**: Validates HTTP 200 and `{"status": "alive"}`.
3. **Readiness probe (`GET /ready`)**: Validates HTTP 200 and loaded model metadata (`model_version`).
4. **Valid benign command (`POST /predict`)**: Submits `git status`, verifying HTTP 200 and a valid label (`ALLOW` / `ASK`).
5. **Valid dangerous command (`POST /predict`)**: Submits `rm -rf /`, verifying HTTP 200 and `DENY` label.
6. **Valid command with context (`POST /predict`)**: Submits command with `cwd` and `lastUserPrompt`.
7. **Invalid command (empty string)**: Submits `{"command": ""}`, verifying HTTP 422 Unprocessable Entity.
8. **Invalid command (whitespace only)**: Submits `{"command": "   "}`, verifying HTTP 422.
9. **Invalid command (type mismatch)**: Submits numeric command `{"command": 12345}`, verifying HTTP 422.
10. **Invalid payload (unknown fields)**: Submits unsupported payload fields, verifying strict schema rejection (HTTP 422).
11. **Non-existent endpoint (`GET /nonexistent`)**: Verifies standard HTTP 404 response.

### Manual curl examples

```bash
# Health probe
curl http://nattech.fib.upc.edu:40XXX0/health

# Readiness probe
curl http://nattech.fib.upc.edu:40XXX0/ready

# Prediction check
curl -X POST http://nattech.fib.upc.edu:40XXX0/predict \
  -H "Content-Type: application/json" \
  -d '{"command": "rm -rf /"}'

# Rejection of empty command (422)
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://nattech.fib.upc.edu:40XXX0/predict \
  -H "Content-Type: application/json" \
  -d '{"command": ""}'
```
