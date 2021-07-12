from abc import ABC, abstractmethod


class Manip(ABC):
    @abstractmethod
    def manipulate(self):
        pass
