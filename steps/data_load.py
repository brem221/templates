import pandas as pd
from typing_extensions import Annotated
from zenml import step, ArtifactConfig
from zenml.logger import get_logger

logger = get_logger(__name__)

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        return pd.read_csv(self.data_path)

@step
def data_loader(
    data_path: str, random_state: int, target: str, is_inference: bool = False
) -> Annotated[pd.DataFrame, "Dataset"]:
    
    ingest_data = IngestData(data_path)
    dataset = ingest_data.get_data()
    
    inference_size = int(len(dataset) * 0.05)
    inference_subset = dataset.sample(inference_size, random_state=random_state)
    
    if is_inference:
        dataset = inference_subset
        #dataset.drop(columns=target, inplace=True)
    else:
        dataset = dataset.drop(inference_subset.index)
    
    dataset.reset_index(drop=True, inplace=True)
    
    logger.info(f"Dataset with {len(dataset)} records loaded!")
    
    return dataset