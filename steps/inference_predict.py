from typing import Any
import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from zenml.client import Client

logger = get_logger(__name__)


@step
def inference_predict(
    model: Any,
    dataset_inf: pd.DataFrame,
) -> Annotated[pd.Series, "predictions"]:
    try:
        predictions = model.predict(dataset_inf)
        predictions = pd.Series(predictions, name="predicted")
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise e