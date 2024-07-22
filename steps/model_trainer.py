from typing import Optional

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from typing_extensions import Annotated
from zenml import ArtifactConfig, step, log_artifact_metadata
from zenml.logger import get_logger
from .model_config import ModelNameConfig
logger = get_logger(__name__)
import logging
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_trainer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> Annotated[
    RegressorMixin, ArtifactConfig(name="gradient_boosting_regressor", is_model_artifact=True)
]:
    try:
        if config.model_name == "GradientBoostingRegressor":
            mlflow.sklearn.autolog()
            model = GradientBoostingRegressor()
            trained_model = model.fit(X_train, y_train)
            #log_artifact_metadata(
                #artifact_name = "GradientBoostingReg",
                #version=None,
                #metadata={"model": trained_model}
           # )
            return trained_model
        else:
            logging.error("Model not found")
            raise ValueError(f"Unknown model type {model}")
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e