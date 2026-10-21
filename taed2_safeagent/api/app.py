from contextlib import asynccontextmanager
import logging
import pickle
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Request

from taed2_safeagent.api.model import Predictor, load_predictor
from taed2_safeagent.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    ServiceResponse,
)

LOGGER = logging.getLogger(__name__)
UNAVAILABLE_RESPONSE = {
    "model": ErrorResponse,
    "description": "The model could not be loaded. Check the model files and restart the service.",
    "content": {"application/json": {"example": {"detail": "Model is unavailable"}}},
}
PREDICTION_EXAMPLES = {
    "inspect_repository": {
        "summary": "Classify git status",
        "value": {"command": "git status"},
    },
    "destructive_command": {
        "summary": "Classify a destructive command without executing it",
        "value": {"command": "rm -rf /"},
    },
    "invalid_input": {
        "summary": "Show input validation with an empty command",
        "value": {"command": ""},
    },
}


def get_predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")
    return predictor


def create_app(loader=None):
    loader = load_predictor if loader is None else loader

    @asynccontextmanager
    async def lifespan(application):
        application.state.predictor = None
        try:
            application.state.predictor = loader()
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            EOFError,
            pickle.UnpicklingError,
            ImportError,
            RuntimeError,
        ) as error:
            LOGGER.error("Model could not be loaded (%s)", type(error).__name__)
        try:
            yield
        finally:
            application.state.predictor = None

    application = FastAPI(
        title="SafeAgent",
        description=(
            "Classify one proposed shell command as ALLOW, ASK or DENY without executing it. "
            "Check /ready to identify the loaded model, then try /predict using the examples. "
            "The deployed SVM uses the command only and returns no confidence score."
        ),
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Service", "description": "Application status and loaded model identity."},
            {"name": "Predictions", "description": "Classify commands without running them."},
        ],
    )

    @application.get(
        "/", response_model=ServiceResponse, tags=["Service"], summary="Identify the service"
    )
    def root():
        return {"service": "SafeAgent"}

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Service"],
        summary="Check whether the application is alive",
        description="This check stays successful even when model loading failed. Use /ready next.",
    )
    def health():
        return {"status": "alive"}

    @application.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["Service"],
        summary="Check model readiness and version",
        responses={503: UNAVAILABLE_RESPONSE},
    )
    def ready(predictor: Annotated[Predictor, Depends(get_predictor)]):
        return {"status": "ready", **predictor.info.model_dump()}

    @application.post(
        "/predict",
        response_model=PredictResponse,
        tags=["Predictions"],
        summary="Classify one shell command",
        description=(
            "Submit one command and optional context. Empty commands, wrong types, batches "
            "and unknown fields are rejected. Commands are limited to 16,384 characters. "
            "Context is limited to 32,768 characters after canonical serialization. "
            "The response identifies the exact model file used for prediction."
        ),
        responses={
            503: UNAVAILABLE_RESPONSE,
            500: {
                "model": ErrorResponse,
                "description": "Inference failed or returned an invalid label.",
                "content": {"application/json": {"example": {"detail": "Prediction failed"}}},
            },
        },
    )
    def predict(
        payload: Annotated[PredictRequest, Body(openapi_examples=PREDICTION_EXAMPLES)],
        predictor: Annotated[Predictor, Depends(get_predictor)],
    ):
        if predictor.info.input_mode == "command-context" and payload.context is None:
            raise HTTPException(status_code=422, detail="Context is required for this model")
        context = payload.context.model_dump() if payload.context is not None else None
        try:
            label = predictor.predict(payload.command, context)
            return PredictResponse(label=label, **predictor.info.model_dump())
        except (ValueError, TypeError, RuntimeError, OSError) as error:
            LOGGER.error("Prediction failed (%s)", type(error).__name__)
            raise HTTPException(status_code=500, detail="Prediction failed") from error

    return application


app = create_app()
