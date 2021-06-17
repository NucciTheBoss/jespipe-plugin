from abc import ABC, abstractmethod


class Predict(ABC):
    @abstractmethod
    def model_predict(self, parameters, model):
        pass
