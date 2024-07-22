from steps.data_load import data_loader
from steps.inference_predict import inference_predict
from steps.inference_preprocessor import inference_preprocessor
from steps.data_splitter import split_data 
from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger
import pandas as pd
from steps.data_preprocessor import data_preprocessor
from steps.data_prep import data_prep
from steps.inference_load import inference_data_loader
logger = get_logger(__name__)

@pipeline(enable_cache=True)
def inference_pipeline(data_path: str, random_state: int, target: str):
    try:
        artifact = Client().get_artifact_version('acd1bb9d-be28-4828-a81f-0a13bf67d1a4')
        loaded_model = artifact.load()

        artifact = Client().get_artifact_version('bd2cddc8-9c3e-4a9c-8de5-c68ca714c4f3')
        loaded_artifact = artifact.load()

        raw_data = inference_data_loader(data_path=data_path, random_state=random_state, target=target)

        #processed_data = inference_preprocessor(dataset_inf=raw_data, target=target)
        X,y, preprocess_pipeline = data_prep(dataset=raw_data)
        X_inf = inference_preprocessor(dataset_inf=raw_data, preprocess_pipeline= loaded_artifact,target=target)

        predictions = inference_predict(
            model=loaded_model,
            dataset_inf=X_inf,
        )
        return predictions

    except Exception as e:
        logger.error(f"Inference pipeline failed: {str(e)}")
        raise e


