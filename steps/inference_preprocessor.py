import pandas as pd
from typing_extensions import Annotated
from zenml.logger import get_logger
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing import Tuple
from zenml import step
import logging
from sklearn.pipeline import Pipeline
from steps.data_preprocessor import data_preprocessor

logger = get_logger(__name__)

@step
def inference_preprocessor(
    dataset_inf: pd.DataFrame,
    preprocess_pipeline: Pipeline,
    target: str,
) -> Annotated[pd.DataFrame, "inference_dataset"]:
    try:
        dataset_inf[target] = pd.Series([1] * dataset_inf.shape[0]) 
        dataset_inf= preprocess_pipeline.transform(dataset_inf)
        if target in dataset_inf.columns:
            dataset_inf.drop(columns=[target], inplace=True)
        return dataset_inf
    except Exception as e:
        logging.error("Error in preprocessing inference data: {}".format(e))
        raise e

