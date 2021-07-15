from jespipe.plugin.start import start
from jespipe.plugin.clean.plotter import Plot


class RMSEMSE(Plot):
    def __init__(self) -> None:
        pass

    def plot(self) -> None:
        pass


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "clean":
        pass

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
