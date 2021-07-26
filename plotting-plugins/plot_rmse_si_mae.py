import math
import random
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
from jespipe.plugin.start import start
from jespipe.plugin.clean.plotter import Plot


class RmseSiMae(Plot):
    def __init__(self, parameters: dict) -> None:
        """
        Create a RmseSiMae plotter instance.

        ### Parameters:
        :param parameters: Parameter dictionary sent by Jespipe

        ### Methods:
        - public
          - plot (abstract): Create plot using data passed by the user.
        """
        self.model_list = parameters["model_list"]
        self.plot_name = parameters["plot_name"]
        self.save_path = parameters["save_path"]

    def plot(self) -> None:
        """
        Create plot using data passed by the user. Makes two separate graphs.
        One graph is for RMSE v.s. MAE over the change budget, the other for Scatter Index
        over the change budget.
        """
        cw_l2_data_list = list(); cw_linf_data_list = list()

        for model in self.model_list:
            cw_l2_data_list.append(joblib.load(model + "/stat/mse-rmse-si-mae-cw_l2_1.pkl"))

        cw_l2_attack = list(zip(self.model_list, cw_l2_data_list))

        for model in self.model_list:
            cw_linf_data_list.append(joblib.load(model + "/stat/mse-rmse-si-mae-cw_inf_1.pkl"))

        cw_linf_attack = list(zip(self.model_list, cw_linf_data_list))

        # RMSE v.s. MAE over change budget
        # There will be one graph for each manipulation
        # CW_L2 ATTACK
        for datum in cw_l2_attack:
            ran_color_list = self._random_color_picker(2)
            fig, axis_1 = plt.subplots()

            # Generate x_axis
            x_axis = list()
            for key in datum[1]:
                if float(key) not in x_axis:
                    x_axis.append(float(key))

            x_axis = x_axis.sort()

            # PLOT RMSE ON AXIS 1
            # Generate y_axis ticks for RMSE
            rmse_values = list()
            for key in datum[1]:
                rmse_values.append(datum[1][key]["rmse"])

            max_rmse = max(rmse_values)
            y_axis_1_ticks = np.linspace(0.0, float(math.ceil(max_rmse)))

            # Plot RMSE
            axis_1.plot(x_axis, rmse_values, color=ran_color_list[0], linestyle="solid")
            axis_1.set_xlabel("Perturbation Budget")
            axis_1.set_ylabel("Root Mean Squared Error (RMSE)", color=ran_color_list[0])
            axis_1.set_yticks(y_axis_1_ticks, labelcolor=ran_color_list[0])

            # PLOT MAE ON AXIS 2
            axis_2 = axis_1.twinx()

            # Generate y-axis ticks for MAE
            mae_values = list()
            for key in datum[1]:
                mae_values.append(datum[1][key]["mae"])

            max_mae = max(mae_values)
            y_axis_2_ticks = np.linspace(0.0, float(math.ceil(max_mae)))

            # Plot MAE
            axis_2.plot(x_axis, mae_values, color=ran_color_list[1], linestyle="solid")
            axis_2.set_ylabel("Mean Absolute Error (MAE)", color=ran_color_list[1])
            axis_2.set_yticks(y_axis_2_ticks, labelcolor=ran_color_list[1])

            fig.tight_layout()

            model_tag = datum[0].split("/"); model_tag = model_tag[-1]
            plt.title("RMSE and MAE as Perturbation Budget increases for CW_L2 attack on model {}".format(model_tag))
            plt.savefig(self.save_path + "/{}-cw_l2-rmse-mae-{}.png".format(model_tag))
            plt.close()

        # CW_Linf ATTACK
        for datum in cw_linf_attack:
            ran_color_list = self._random_color_picker(2)
            fig, axis_1 = plt.subplots()
            # TODO: Use the twinx method to add a second y axis for RMSE v.s. MAE
            axis_2 = axis_1.twinx()
            
            plt.close()
        
        # Scattter Index over the change budget
        # All the manipulations will be put on the same graph.
        # CW_L2 ATTACK
        plt.figure()
        plt.xlabel("Perturbation Budget"); plt.ylabel("Scatter Index (in %)")
        plt.title("Scatter Index as Perturbation Budget increases for CW_L2 attack")
        ran_color_list = self._random_color_picker(len(cw_l2_attack)); i = 0

        # Find maximum scatter index value
        scatter_values = list()
        for datum in cw_l2_attack:
            for key in datum[1]:
                scatter_values.append(datum[1][key]["scatter_index"])

        max_scatter = max(scatter_values)

        # Generate y_axis ticks
        y_axis_ticks = np.linspace(0.0, float(math.ceil(max_scatter)))
        plt.yticks(y_axis_ticks)

        # Generate x_axis
        x_axis = list()
        for datum in cw_l2_attack:
            for key in datum[1]:
                if float(key) not in x_axis:
                    x_axis.append(float(key))

        x_axis = x_axis.sort()

        for datum in cw_l2_attack:
            values = list()
            for key in datum[1]:
                values.append(datum[1][key]["scatter_index"])

            # Append values to the plot
            line_name = datum[0].split("/"); line_name = line_name[-1]
            plt.plot(x_axis, values, color=ran_color_list[i], linestyle=self._random_linestyle(), label=line_name)
            i += 1

        plt.legend()
        plt.savefig(self.save_path + "/{}-cw_l2-sci-pertur_budget.png".format(self.plot_name))
        plt.close()

        # CW_Linf ATTACK
        plt.figure()
        plt.xlabel("Perturbation Budget"); plt.ylabel("Scatter Index (in %)")
        plt.title("Scatter Index as Perturbation Budget increases for CW_Linf attack")
        ran_color_list = self._random_color_picker(len(cw_linf_attack)); i = 0

        # Find maximum scatter index value
        scatter_values = list()
        for datum in cw_linf_attack:
            for key in datum[1]:
                scatter_values.append(datum[1][key]["scatter_index"])
        
        max_scatter = max(scatter_values)

        # Generate y_axis ticks
        y_axis_ticks = np.linspace(0.0, float(math.ceil(max_scatter)))
        plt.yticks(y_axis_ticks)

        # Generate x_axis
        x_axis = list()
        for datum in cw_l2_attack:
            for key in datum[1]:
                if float(key) not in x_axis:
                    x_axis.append(float(key))

        x_axis = x_axis.sort()

        for datum in cw_linf_attack:
            values = list()
            for key in datum[1]:
                values.append(datum[1][key]["scatter_index"])

            # Append values to the plot
            line_name = datum[0].split("/"); line_name = line_name[-1]
            plt.plot(x_axis, values, color=ran_color_list[i], linestyle=self._random_linestyle(), label=line_name)
            i += 1

        plt.legend()
        plt.savefig(self.save_path + "/{}-cw_linf-sci-perturb_budget.png".format(self.plot_name))
        plt.close()

    def _random_color_picker(self, num_of_categories: int) -> List[str]:
        """
        Internal method for randomly assigning hexadecimal colors to
        the categories of a plot.

        ### Parameters:
        :param num_of_categories: Number of categories that are going to be in the plot.

        ### Returns:
        :return: List of hexadecimal colors.
        """
        color_list = list()

        i = 0
        while i < num_of_categories:
            ran_color = f"#{random.randrange(0x100000):06x}"
            if ran_color not in color_list:
                color_list.append(ran_color); i += 1

        return ran_color

    def _random_linestyle(self) -> str:
        """
        Internal method for selecting a random linestyle in Matplotlib

        ### Returns:
        :return: solid, dotted, dashed, or dashdot linestyle.
        """
        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        i = random.randint(0, 3)
        return linestyles[i]


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "clean":
        plotter = RmseSiMae(parameters)
        plotter.plot()

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
