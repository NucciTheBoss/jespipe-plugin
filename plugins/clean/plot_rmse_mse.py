import matplotlib.pyplot as plt
import numpy as np
from jespipe.plugin.start import start
from jespipe.plugin.clean.plotter import Plot


class RMSEMSE(Plot):
    def __init__(self, parameters: dict) -> None:
        """
        Create a RMSEMSE plotter instance.

        ### Parameters:
        :param parameters: Parameter dictionary sent by Jespipe

        ### Methods:
        - public
          - plot (abstract): Create plot using data passed by the user.
        """
        self.plot_name = parameters["plot_name"]
        self.tag_list = parameters["tag_list"]
        self.data_list = parameters["data_list"]
        self.save_path = parameters["save_path"]

    def plot(self) -> None:
        """
        Create plot using data passed by the user. Makes two separate graphs.
        One graph is for mean squared error, the other for root mean squared
        error.
        """
        # Plot Mean Squared Error

        # Plot Root Mean Squared Error


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "clean":
        pass

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
