import pandas as pd
import pytest

from taed2_safeagent.modeling.registry import input_texts


def test_registry_keeps_context_out_of_command_inputs():
    data = pd.DataFrame([{"command": "git status", "context": '{"cwd": "/repo"}'}])
    assert input_texts(data, "command") == ["git status"]
    assert '"/repo"' in input_texts(data, "command-context")[0]


@pytest.mark.parametrize(
    "row",
    [
        {"label": "ALLOW"},
        {"command": "git status", "reason": "safe"},
        {"command": None},
        {"command": "git status", "context": "{broken"},
    ],
)
def test_registry_rejects_invalid_inputs(row):
    with pytest.raises((ValueError, TypeError)):
        input_texts(pd.DataFrame([row]), "command-context")
