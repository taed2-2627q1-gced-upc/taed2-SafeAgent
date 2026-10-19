# Virtech deployment and runtime guide

This guide documents the setup, runtime environment, and model deployment on
**Virtech** (the private OpenNebula cloud of the Facultat d'Informàtica de Barcelona, UPC).
It details how to provision the virtual machine, configure DVC remote authentication
without committing secrets, install frozen dependencies, and serve the SafeAgent API.

## Infrastructure and network architecture

Virtech hosts Linux virtual machines with network virtualization managed through NAT.
When selecting a template with **sense VPN** (external NAT ports), the machine is accessible
from the internet without needing the UPClink VPN.

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
4. Run `dvc pull models/tfidf_char_linear_svm` to retrieve the trained model weights.
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

Then run `uv run --frozen --group data dvc pull models/tfidf_char_linear_svm`.

## Step 4: Running the API service

Run the Uvicorn server bound to all interfaces on port `8080`:

```bash
bash scripts/run_api.sh
```

Or manually:

```bash
uv run --frozen uvicorn taed2_safeagent.api.app:app --host 0.0.0.0 --port 8080
```

### Running as a background service
To keep the service running after disconnecting from SSH, use `nohup` or `tmux`:

```bash
nohup bash scripts/run_api.sh > api.log 2>&1 &
```

To stop or restart:

```bash
# Find and terminate process
pkill -f uvicorn
# Restart
nohup bash scripts/run_api.sh > api.log 2>&1 &
```

## Step 5: Verification and smoke testing

From any external machine connected to the internet (no VPN required):

### Health check
```bash
curl http://nattech.fib.upc.edu:40XXX0/health
```
Expected response:
```json
{"status":"healthy"}
```

### Readiness check
```bash
curl http://nattech.fib.upc.edu:40XXX0/ready
```
Expected response:
```json
{"status":"ready","model":"tfidf_char_linear_svm","model_version":"sha256:...","input_mode":"command"}
```

### Prediction smoke test
```bash
curl -X POST http://nattech.fib.upc.edu:40XXX0/predict \
  -H "Content-Type: application/json" \
  -d '{"command": "rm -rf /"}'
```
Expected response:
```json
{"label":"DENY","model":"tfidf_char_linear_svm","model_version":"sha256:...","input_mode":"command"}
```

