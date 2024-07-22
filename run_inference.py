from pipelines.inference import inference_pipeline

if __name__ == "__main__":
    data_path = "data/inference_subset.csv"
    random_state = 42
    target = "aveOralM"

    inference_pipeline(data_path=data_path, random_state=random_state, target=target)
