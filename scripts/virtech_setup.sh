#!/usr/bin/env bash
# Virtech VM Setup and Runtime Provisioning
# Prepares the frozen environment and recovers the selected DVC model
# without storing any credentials in Git.

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"

echo "=== 1. Checking uv installation ==="
if ! command -v uv &>/dev/null; then
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
echo "uv version: $(uv --version)"

echo "=== 2. Installing locked environment dependencies ==="
uv sync --frozen --group data

echo "=== 3. Configuring DVC remote credentials (outside Git) ==="
# Check if local credentials or environment variables are present
if [ -n "${DAGSHUB_TOKEN:-}" ]; then
    echo "Configuring DVC storage credentials from DAGSHUB_TOKEN..."
    uv run --frozen --group data dvc remote modify --local storage access_key_id "$DAGSHUB_TOKEN"
    uv run --frozen --group data dvc remote modify --local storage secret_access_key "$DAGSHUB_TOKEN"
elif [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    echo "Using AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
elif [ -f ".dvc/config.local" ] && grep -q "access_key_id" ".dvc/config.local"; then
    echo "Found existing .dvc/config.local credentials."
else
    echo "No DVC credentials detected."
    echo "Please provide your DagsHub token (it will be saved to .dvc/config.local, ignored by Git):"
    read -r -s -p "DagsHub Token: " USER_TOKEN
    echo ""
    if [ -n "$USER_TOKEN" ]; then
        uv run --frozen --group data dvc remote modify --local storage access_key_id "$USER_TOKEN"
        uv run --frozen --group data dvc remote modify --local storage secret_access_key "$USER_TOKEN"
    else
        echo "Error: No token provided. Cannot authenticate with DVC storage remote."
        exit 1
    fi
fi

echo "=== 4. Recovering the model artifact via DVC ==="
uv run --frozen --group data dvc pull models/vm_linear_svm

echo "=== 5. Verifying model runtime loading ==="
uv run --frozen python -c "
from taed2_safeagent.api.model import load_predictor
predictor = load_predictor()
print(f'Predictor loaded successfully: {predictor.info.model} (version: {predictor.info.model_version})')
"

echo "=== Setup complete! ==="
echo "You can now run the API with: bash scripts/run_api.sh"

