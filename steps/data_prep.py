from zenml import step
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
import pandas as pd 
from typing import Tuple
from typing_extensions import Annotated
from sklearn.impute import SimpleImputer

@step
def data_prep(dataset: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X"],
    Annotated[pd.Series, "y"],
    Annotated[Pipeline, "Preprocess_pipeline"]
]:
    try:
        X = dataset.drop(["aveOralM"], axis=1)
        y = dataset["aveOralM"]

        imputer = SimpleImputer(strategy='most_frequent')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        if y.isnull().sum() > 0:
            y.fillna(y.median(), inplace=True)
        categorical_columns = dataset.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_columns, prefix=categorical_columns)
        scaler = MinMaxScaler()
        pca = PCA()
        mutual_info = mutual_info_regression(X,y)
        select = SelectKBest(mutual_info_regression,k=10)
        preprocess_pipeline = Pipeline([
            ("scaler",scaler),
            ("pca",pca),
            ("select",select)
        ])
        X = preprocess_pipeline.fit_transform(X,y)
        features = pd.DataFrame(X)

        return features, y, preprocess_pipeline
    except Exception as e:
        raise e