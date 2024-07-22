from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step
from src.data_cleaning import DataSplitStrategy, DataCleaning
import logging

@step
def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[Annotated[pd.DataFrame, "X_train"],
                                                       Annotated[pd.DataFrame, "X_test"],
                                                       Annotated[pd.Series, "y_train"],
                                                       Annotated[pd.Series, "y_test"]]:
    try:
        split_strategy = DataSplitStrategy()
        X_train, X_test, y_train, y_test = split_strategy.handle_data(X,y)
        logging.info("Data Spliting Completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e

