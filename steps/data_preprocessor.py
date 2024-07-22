import pandas as pd
from typing_extensions import Annotated
from zenml.logger import get_logger
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing import Tuple
from zenml import step
import logging
import joblib
logger = get_logger(__name__)

@step
def data_preprocessor(
    dataset: pd.DataFrame,
    save_path: str = "preprocessor.pkl"
) -> Tuple[
    Annotated[pd.DataFrame, "X"], 
    Annotated[pd.Series, "y"]
]:
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(dataset, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X, y = data_cleaning.handle_data()

        joblib.dump(data_cleaning, save_path)
        logging.info(f"Preprocessing steps saved to {save_path}")
        logging.info("Data division completed")
        return X, y
    except Exception as e:
        logging.error("Error in dividing data: {}".format(e))
        raise e
    


