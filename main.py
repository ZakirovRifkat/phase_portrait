import numpy as np
from phase_portrait import InitialCondition, PhasePortrait


def system(t, state):
    """
    Defines a system of differential equations.

    Args:
        t (float): Time variable (not used in the computation but required for ODE solvers).
        state (list of float): A list containing state variables [x, y].

    Returns:
        list of float: The derivatives [dx/dt, dy/dt] of the system.
    """
    x, y = state
    return [y, -x - y + x**2 + y**2]


initial_conditions_list = [
    InitialCondition([0.5, -1.5]),
    InitialCondition([-0.3, -1.5]),
    InitialCondition([-0.409, -1.5]),
    InitialCondition([0.9, -1.5]),
    InitialCondition([-0.49, -1.5]),
    InitialCondition([1.5, -1.5]),
    InitialCondition([1.64, -1.5]),
    InitialCondition([1.9, -1.5]),
    # reverse time
    InitialCondition([1, 0.5], t_eval=np.linspace(50, 0, 1500), color="red"),
]


phase_portait = PhasePortrait(system=system)
phase_portait.plot_trajectories(
    initial_conditions_list, t_eval=np.linspace(0, 50, 1500)
)
phase_portait.plot_equilibrium(coordinates=(0, 0), is_stable=True)
phase_portait.plot_equilibrium(coordinates=(1, 0), is_stable=False)
phase_portait.show(axis=[-1, 1.9, -1.4, 1.5])
phase_portait.save()
