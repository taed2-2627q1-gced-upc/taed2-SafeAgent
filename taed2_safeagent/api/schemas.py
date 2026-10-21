from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from taed2_safeagent.data.common import canonical_json

MAX_COMMAND_CHARS = 16_384
MAX_CONTEXT_CHARS = 32_768


class StrictSchema(BaseModel):
    """Reject unknown fields and type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class GitStatus(StrictSchema):
    """Describe the three supported file states."""

    untracked: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)
    staged: list[str] = Field(default_factory=list)


class CommandContext(StrictSchema):
    """Accept only the context used by the models."""

    gitRemote: str | None = None
    gitStatus: GitStatus = Field(default_factory=GitStatus)
    agentTouchedFiles: list[str] = Field(default_factory=list)
    lastUserPrompt: str | None = None
    cwd: str | None = None


class PredictRequest(StrictSchema):
    """Classify one command with optional execution context."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={"examples": [{"command": "git status"}]},
    )
    command: str = Field(
        min_length=1,
        max_length=MAX_COMMAND_CHARS,
        description="One proposed shell command. The service never executes it.",
    )
    context: CommandContext | None = Field(
        default=None,
        description="Optional session context. The deployed command only SVM ignores it.",
    )

    @field_validator("command")
    @classmethod
    def nonempty_command(cls, value):
        if not value.strip():
            raise ValueError("Command must contain text")
        return value

    @field_validator("context")
    @classmethod
    def bounded_context(cls, value):
        if value is not None and len(canonical_json(value.model_dump())) > MAX_CONTEXT_CHARS:
            raise ValueError("Context is too long")
        return value


class ModelInfo(StrictSchema):
    """Identify the loaded predictor."""

    model: str = Field(min_length=1, description="Name of the loaded model.")
    model_version: str = Field(
        min_length=1, description="SHA256 of the loaded pipeline.joblib with a sha256: prefix."
    )
    input_mode: Literal["command", "command-context"] = Field(
        description="Inputs used by the loaded model."
    )


class PredictResponse(ModelInfo):
    """Return a prediction without a fabricated confidence score."""

    label: Literal["ALLOW", "ASK", "DENY"] = Field(
        description="Predicted class. The calling agent decides whether to execute the command."
    )


class ServiceResponse(StrictSchema):
    """Identify the application."""

    service: Literal["SafeAgent"]


class HealthResponse(StrictSchema):
    """Describe process liveness independently of model availability."""

    status: Literal["alive"]


class ReadyResponse(ModelInfo):
    """Identify the model available for inference."""

    status: Literal["ready"]


class ErrorResponse(StrictSchema):
    """Describe a service failure without internal details."""

    detail: str
