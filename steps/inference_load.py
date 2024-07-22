from zenml import step
import numpy as np
import pandas as pd
from steps.data_load import IngestData

@step(enable_cache=False)
def inference_data_loader(data_path: str, random_state: int, target:str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data