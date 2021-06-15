from abc import ABC, abstractmethod


class Build(ABC):
    @abstractmethod
    def build_model(self, dataset_path, parameters):
        pass
