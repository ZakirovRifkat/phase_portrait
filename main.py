import numpy as np
from phase_portrait import PhasePortrait


def system(t, y):
    return [y[1], -y[0] - y[1] + y[0] ** 2 + y[1] ** 2]


t_span = (0, 50)
t_eval = np.linspace(0, 50, 1500)


initial_conditions_list = [
    [0.5, -1.5],
    [-0.3, -1.5],
    [-0.407, -1.5],
    [0.9, -1.5],
    [-0.49, -1.5],
    [1.5, -1.5],
    [1.64, -1.5],
    [1.9, -1.5],
]

config = {
    "color": "black",
    "number_of_arrow": 2,
    "arrow_span": 20,
    "arrow_color": "black",
    "arrow_shift": 5,
}


phase_portait = PhasePortrait(system)
phase_portait.plot_state_equilibrium(x=[0, 1], y=[0, 0])
phase_portait.plot_trajectories(initial_conditions_list, t_span, t_eval, config)
phase_portait.show(xlim=(-1.0, 1.9), ylim=(-1.4, 1.5))
phase_portait.save()
