from steps.data_load import data_loader
from steps.data_preprocessor import data_preprocessor
from zenml import pipeline
from zenml.logger import get_logger
from steps.data_prep import data_prep

logger = get_logger(__name__)

@pipeline
def feature_engineering(
    data_path: str,
    target: str,
    random_state: int,
):
    raw_data = data_loader(data_path, random_state=random_state, target=target)
    X, y, preprocess_pipeline = data_prep(dataset=raw_data)
    return X,y, preprocess_pipeline
