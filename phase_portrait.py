from typing import List, Optional, TypedDict, Union
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp


class PlotConfig(TypedDict, total=False):
    color: str
    number_of_arrow: int
    arrow_span: int
    arrow_color: str


class PhasePortrait:
    def __init__(self, system, figsize=(8, 6)):
        """
        Initialize the PhasePortrait object.

        Args:
            system: Callable that describes the system of differential equations.
            figsize: Tuple specifying the figure size for matplotlib.
        """
        self.system = system
        self._default_images_dir = "images"
        self._default_config = {
            "color": "black",
            "number_of_arrow": 1,
            "arrow_span": 10,
            "arrow_color": "black",
        }
        plt.figure(figsize=figsize)

    def plot_trajectory(
        self,
        solution,
        color="black",
        number_of_arrow=1,
        arrow_span=10,
        arrow_color="black",
    ):
        """
        Plot the solution trajectory with optional arrows to indicate direction.

        Args:
            solution: Array of x and y solutions from solve_ivp.
            color: Line color of the trajectory.
            number_of_arrow: Number of arrows to indicate direction.
            arrow_span: Interval between arrows along the trajectory.
            arrow_color: Color of the arrows.
        """
        x, y = solution
        plt.plot(x, y, color=color)
        idx = arrow_span
        for _ in range(number_of_arrow):
            if idx >= len(x):
                continue
            plt.quiver(
                x[idx],
                y[idx],
                x[idx + 5] - x[idx],
                y[idx + 5] - y[idx],
                color=arrow_color,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.008,
                zorder=2,
            )
            idx += arrow_span

    def plot_state_equilibrium(self, x, y, color="blue"):
        """
        Plot equilibrium states as scatter points.

        Args:
            x: X-coordinate of the equilibrium state.
            y: Y-coordinate of the equilibrium state.
            color: Color of the equilibrium points.
        """
        plt.scatter(x, y, s=50, c=color, zorder=3)

    def plot_trajectories(
        self,
        initial_conditions_list,
        t_span,
        t_eval,
        config: Optional[Union[PlotConfig, List[PlotConfig]]] = None,
    ):
        """
        Visualize the trajectories for a list of initial conditions.

        Args:
            initial_conditions_list: List of initial conditions [[x0, v0], ...].
            t_span: Tuple (t_start, t_end) specifying the time interval for integration.
            t_eval: Array of time points where the solution is evaluated.
            config: Configuration for plotting trajectories. Can be:
                - A single dictionary applied to all trajectories.
                - A list of dictionaries, one for each trajectory.
        """
        if config is None:  # Use default configuration if not provided
            config = self._default_config

        if isinstance(config, dict):
            config = [config] * len(initial_conditions_list)
        elif isinstance(config, list):
            if len(config) != len(initial_conditions_list):
                raise ValueError(
                    "The length of the list of configurations must match the number of initial conditions."
                )

        for initial_conditions, cfg in zip(initial_conditions_list, config):
            sol = solve_ivp(
                self.system, t_span, initial_conditions, t_eval=t_eval, method="RK45"
            )
            self.plot_trajectory(
                solution=[sol.y[0], sol.y[1]],
                color=cfg["color"],
                number_of_arrow=cfg["number_of_arrow"],
                arrow_span=cfg["arrow_span"],
                arrow_color=cfg["arrow_color"],
            )

    def show(
        self,
        xlim=None,
        ylim=None,
        xlabel="x",
        ylabel="y",
        title="Phase portrait",
        legend="",
    ):
        """
        Display the plotted phase portrait.

        Args:
            xlim: Tuple (x_min, x_max) for x-axis limits.
            ylim: Tuple (y_min, y_max) for y-axis limits.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title of the plot.
            legend: Legend for the plot.
        """
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(legend)
        plt.title(title)
        plt.show()

    def save(self, path=None):
        """
        Save the plotted phase portrait to a file.

        Args:
            path: Directory where the plot will be saved. Defaults to "images".
        """
        idx = 0
        if path is None:
            path = self._default_images_dir
        if os.path.exists(path):
            idx = len(os.listdir(path)) + 1  # Increment file index
        else:
            os.mkdir(path)
        image_path = os.path.join(path, f"plot_{idx}.pdf")
        plt.savefig(image_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to {image_path}")
