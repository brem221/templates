import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import GradientBoostingRegressor

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class GradientBoostingModel(Model):
    def train(self, X_train, y_train):
        
        try:
            reg = GradientBoostingRegressor()
            reg.fit(X_train, y_train)
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))