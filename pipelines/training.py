from zenml import pipeline
from steps.data_load import data_loader
from steps.data_preprocessor import data_preprocessor
from steps.data_splitter import split_data
from steps.model_trainer import model_trainer
from steps.model_evaluator import evaluate_model
from pipelines.feature_engineering import feature_engineering

@pipeline(enable_cache=True)
def training_pipeline(data_path, random_state:int, target: str):
    features,y, preprocess_pipeline = feature_engineering(data_path=data_path, random_state=random_state,target= target)
    X_train, X_test, y_train, y_test = split_data(features, y)
    model = model_trainer(X_train,X_test, y_train, y_test)
    r2, mse, rmse = evaluate_model(model, X_test, y_test)