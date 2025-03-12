import os
from typing import List, Optional, Union, TypedDict
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


DEFAULT_CONFIG = {
    "arrow_span": 20,
    "number_of_arrow": 1,
}


class PlotConfig(TypedDict, total=False):
    arrow_span: int
    number_of_arrow: int


class InitialCondition:
    def __init__(
        self,
        point,
        t_eval=None,
        rtol=1e-6,
        atol=1e-10,
        color="black",
        arrow_color="black",
        config: Optional[Union[PlotConfig, List[PlotConfig]]] = None,
    ):
        self.point = point
        self.t_eval = t_eval
        self.rtol = rtol
        self.atol = atol
        self.color = color
        self.arrow_color = arrow_color
        if config is None:
            config = DEFAULT_CONFIG
        self.config = config


class PhasePortrait:
    def __init__(self, system):
        """
        Initialize the PhasePortrait object.

        Args:
            system: Callable that describes the system of differential equations.
        """
        self.system = system
        self.fig, self.ax = plt.subplots()

    def plot_trajectory(
        self,
        solution,
        color="black",
        arrow_span=20,
        arrow_color="black",
        number_of_arrow=1,
        t_eval=None,
    ):
        """
        Plot the solution trajectory with optional arrows to indicate direction.

        Args:
            solution: Array of x and y solutions from solve_ivp.
            color: Line color of the trajectory.
            arrow_span: Interval between arrows along the trajectory.
            arrow_color: Color of the arrows.
            number_of_arrow: Number of arrows
        """
        is_reverse = False
        if not t_eval is None and t_eval[0] > t_eval[-1]:
            is_reverse = True

        x, y = solution
        self.ax.plot(x, y, color=color)

        idx = arrow_span
        for _ in range(number_of_arrow):
            xy = (x[idx], y[idx])
            xytext = (x[idx - 2], y[idx - 2])

            if is_reverse:
                temp = xy
                xy = xytext
                xytext = temp

            if idx >= len(x):
                continue
            self.ax.annotate(
                "",
                xy=xy,
                xytext=xytext,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": arrow_color,
                    "lw": 1.2,
                    "mutation_scale": 20,
                },
            )
            idx += arrow_span

    def plot_equilibrium(self, coordinates, is_stable=False):
        """
        Plot equilibrium states as scatter points.

        Args:
            coordinates: Tuple of coordinates (x, y).
            is_stable: Whether the equilibrium point is stable.
        """
        color = "green" if is_stable else "red"
        self.ax.scatter(coordinates[0], coordinates[1], color=color, s=50, zorder=2)

    def plot_trajectories(
        self,
        initial_conditions_list: List[InitialCondition],
        t_eval=None,
    ):
        """
        Visualize the trajectories for a list of initial conditions.

        Args:
            initial_conditions_list: List of initial conditions [[x0, v0], ...].
            t_span: Tuple (t_start, t_end) specifying the time interval for integration.
            t_eval: List of time evaluation points for each trajectory.
        """
        for initial_condition in initial_conditions_list:

            if t_eval is None and initial_condition.t_eval is None:
                raise ValueError("Need to provide t_eval value")

            if initial_condition.t_eval is None:
                initial_condition.t_eval = t_eval

            sol = solve_ivp(
                self.system,
                y0=initial_condition.point,
                t_span=[initial_condition.t_eval[0], initial_condition.t_eval[-1]],
                t_eval=initial_condition.t_eval,
                rtol=initial_condition.rtol,
                atol=initial_condition.atol,
                method="RK45",
            )

            self.plot_trajectory(
                solution=[sol.y[0], sol.y[1]],
                color=initial_condition.color,
                arrow_span=initial_condition.config["arrow_span"],
                arrow_color=initial_condition.arrow_color,
                number_of_arrow=initial_condition.config["number_of_arrow"],
                t_eval=initial_condition.t_eval,
            )

    def show(self, xlabel="x", ylabel="y", title="Phase Portrait", axis=None):
        """
        Display the plotted phase portrait.

        Args:
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title of the plot.
            axis: Iterable object of (x_min, x_max, y_min, y_max) to set axis limits.
        """
        self.ax.set_title(title)

        self.ax.set_xlabel(xlabel, fontsize=16)
        self.ax.set_ylabel(ylabel, fontsize=16, rotation=0)
        self.ax.tick_params(labelsize=14)
        self.ax.tick_params(labelsize=14)

        self.ax.grid(True)

        if axis is not None:
            self.ax.set_xlim([axis[0], axis[1]])
            self.ax.set_ylim([axis[2], axis[3]])

        self.ax.legend()
        plt.show()

    def save(self, path=None):
        """
        Save the plotted phase portrait to a file.

        Args:
            path: Directory where the plot will be saved. Defaults to "images".
        """
        if path is None:
            path = "images"
        if not os.path.exists(path):
            os.makedirs(path)
        file_count = len(
            [
                name
                for name in os.listdir(path)
                if name.startswith("plot_") and name.endswith(".pdf")
            ]
        )
        image_path = os.path.join(path, f"plot_{file_count + 1}.pdf")
        self.fig.savefig(image_path, dpi=300)
        print(f"Figure saved to {image_path}")
