from taed2_safeagent.data.common import CONTEXT_FIELDS, STATUS_FIELDS, canonical_json, digest


def _path_list(value):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Expected a list of file paths")
    return sorted(set(value))


def safe_context(value):
    if not isinstance(value, dict):
        raise TypeError("Expected a context object")
    for field in ("gitRemote", "gitStatus", "agentTouchedFiles"):
        if field not in value:
            raise ValueError(f"Missing context field: {field}")
    status = value["gitStatus"]
    if not isinstance(status, dict) or set(status) != set(STATUS_FIELDS):
        raise ValueError("Invalid Git status fields")
    result = {field: value.get(field) for field in CONTEXT_FIELDS}
    for field in ("gitRemote", "lastUserPrompt", "cwd"):
        if result[field] is not None and not isinstance(result[field], str):
            raise ValueError(f"Expected a string or null: {field}")
    result["gitStatus"] = {field: _path_list(status[field]) for field in STATUS_FIELDS}
    result["agentTouchedFiles"] = _path_list(value["agentTouchedFiles"])
    return result


def informative_context(context):
    return bool(
        any(context["gitStatus"].values())
        or context["agentTouchedFiles"]
        or context["lastUserPrompt"]
        or context["cwd"]
    )


def example_id(command, context):
    return digest(canonical_json([command, context]))


def build_input(example, mode):
    command = example["command"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Expected a nonempty command")
    if mode == "command":
        return command
    if mode == "command-context":
        context = safe_context(example["context"])
        return f"COMMAND\n{command}\nCONTEXT\n{canonical_json(context)}"
    raise ValueError(f"Unknown input mode: {mode}")
