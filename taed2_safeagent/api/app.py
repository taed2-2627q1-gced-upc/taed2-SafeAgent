from contextlib import asynccontextmanager
import logging
import pickle
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from taed2_safeagent.api.model import Predictor, load_predictor
from taed2_safeagent.api.schemas import PredictRequest, PredictResponse

LOGGER = logging.getLogger(__name__)


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

    application = FastAPI(title="SafeAgent", lifespan=lifespan)

    @application.get("/")
    def root():
        return {"service": "SafeAgent"}

    @application.get("/health")
    def health():
        return {"status": "alive"}

    @application.get("/ready")
    def ready(predictor: Annotated[Predictor, Depends(get_predictor)]):
        return {"status": "ready", **predictor.info.model_dump()}

    @application.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, predictor: Annotated[Predictor, Depends(get_predictor)]):
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
