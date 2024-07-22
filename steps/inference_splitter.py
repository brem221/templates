from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step
from src.data_cleaning import DataSplitStrategy
import logging

@step
def split_data(X: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"]
]:
    try:
        split_strategy = DataSplitStrategy()
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        logging.info("Data Splitting Completed")
        return X_train, X_test
    except Exception as e:
        logging.error(f"Error in splitting data: {str(e)}")
        raise e
