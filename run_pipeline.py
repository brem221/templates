from pipelines.feature_engineering import feature_engineering
from pipelines.training import training_pipeline
from zenml.client import Client

# from pipelines.inference import inference_pipeline

if __name__ == "__main__":
    data_path = "data/data.csv"
    target = "aveOralM"
    random_state = 42
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    feature_engineering(data_path, random_state=random_state, target=target)

    training_pipeline(data_path=data_path, random_state=random_state, target=target)

    # inference_pipeline(data_path=data_path,random_state=random_state, target=target)
