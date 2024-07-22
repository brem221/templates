from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, Tuple
import logging
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, dataset: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    def handle_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        try:
            if dataset is None:
                raise ValueError("Data is None")
            dataset = dataset.dropna()
            categorical_columns = dataset.select_dtypes(include=['object']).columns
            for column in categorical_columns:
                dataset = pd.get_dummies(dataset, columns=[column], prefix=[column])
            return dataset
        except Exception as e:
            logging.error("Error while preprocessing data: {}".format(e))
            raise e
        
class DataDivideStrategy(DataStrategy):
    def handle_data(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            if dataset is None:
                raise ValueError("Data is None")
            X = dataset.drop(["aveOralM"], axis=1)
            y = dataset["aveOralM"]
            return X, y
        except Exception as e:
            logging.error("Error while dividing data: {}".format(e))
            raise e
        
class DataSplitStrategy:
    def handle_data(self, X: pd.DataFrame, y = pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error while splitting data: {}".format(e))
            raise e

class DataCleaning:
    def __init__(self, dataset: pd.DataFrame, strategy: DataStrategy):
        if dataset is None:
            raise ValueError("Data is None")
        self.dataset = dataset
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple]:
        try:
            return self.strategy.handle_data(self.dataset)
        except Exception as e:
            logging.error("Error while handling data: {}".format(e))
            raise e
        
if __name__ == "__main__":
    try:
        data_path = "data/data.csv"
        dataset = pd.read_csv(data_path)

        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(dataset, preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X, y = data_cleaning.handle_data()

        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(X,y, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        print("Data preprocessing, division and splitting completed")
        print("X:", X.head())
        print("Y:", y.head())
        print("X_train:", X_train.head())
        print("X_test:", X_test.head())
        print("y_train:", y_train.head())
        print("y_test:", y_test.head())
    except Exception as e:
        logging.error("Error in the data pipeline: {}".format(e))
