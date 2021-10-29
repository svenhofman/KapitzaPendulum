import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os

from runge_kutta_4 import Parameters
from helper_functions import aux_runge_kutta_4
from helper_functions import is_equilibrium
from helper_functions import make_gif

# Global variables that correspond to Top, Bottom and No equlibrium
OTHER = -1
BOTTOM = 0
TOP = 1


# Saves the data in the data folder with corresponding name
def save_data(arr, name):
    np.savetxt(os.path.join("data", name), arr, delimiter=",")

# Simulation corresponding to Figure 11
def grid_simulation():
    # Initialise the parameters and ranges
    h = 0.01
    t_end = 50
    A_max = 300
    omega_max = 40
    n_values_A = 150
    n_values_omega = 50

    # Grid for A from 1 to 300, 150 values
    A_range = np.linspace(1, A_max, n_values_A)
    # Grid for omega from 1 to 40, 50 values
    omega_range = np.linspace(1, omega_max, n_values_omega)

    res = np.empty((len(omega_range), len(A_range)))

    row, col = 0, 0
    # Rows are different omega's corresponding to same A
    for A in A_range:
        for omega in omega_range:
            # Solve the system
            sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h))
            pos = is_equilibrium(sol)[1]
            print(f"A: {A} ({col}/{len(A_range)}), omega: {omega} ({row}/{len(omega_range)}) => Stability: {pos}")
            res[row, col] = pos
            row = row + 1
        row = 0
        col = col + 1
    save_data(res,
             f"A_range[1,{A_max}]({n_values_A})_omega_range[1,{omega_max}]({n_values_omega})_h={h}_t_end={t_end}.csv")

# Simulation corresponding to Figure 9
def smallest_omega_stability_simulation():
    # Initialise the parameters and ranges
    h = 0.01
    t_end = 10
    A_min = 1
    A_max = 500 
    omega_min = 1
    omega_max = 50
    A_step = 1
    omega_step = 1
    A_range = np.arange(A_min, A_max + A_step / 2, A_step)
    omega_range = np.arange(omega_min, omega_max + omega_step / 2, omega_step)

    res = np.empty((2, len(A_range)))
    col = 0
    for A in A_range:
        # First row corresponds to amplitude
        res[0, col] = A
        # By default the value of omega is -1
        res[1, col] = -1
        for omega in omega_range:
            # Solve the system
            sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h))
            # If the top stability is reached, set row value to corresponding omega
            if is_equilibrium(sol)[1] == TOP:
                res[1][col] = omega
                break
        col += 1
    save_data(res, "smallest_omega_stability_simulation.csv")

# Simulation corresponding to Figure 10
def regression_smallest_stability_omega_simulation():
    # Initialise the parameters and ranges
    h = 0.01
    t_end = 10
    A_min = 200
    A_max = 500
    omega_min = 20
    omega_max = 40
    A_step = .1
    omega_step = .0001
    A_range = np.arange(A_min, A_max + A_step / 2, A_step)
    omega_range = np.arange(omega_min, omega_max + omega_step / 2, omega_step)

    res = np.empty((2, len(A_range)))
    col = 0
    min_idx = 0

    for A in A_range:
        # First row corresponds to amplitude
        res[0, col] = A
        # By default the value of omega is -1
        res[1, col] = -1
        idx = min_idx
        while idx < len(omega_range):
            omega = omega_range[idx]
            # Solve the system
            sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h))
            print(f"A: {A} ({col}/{len(A_range)}), omega: {omega}")
            # If the top stability is reached, set row value to corresponding omega
            if is_equilibrium(sol)[1] == TOP:
                print("=> Stability: TOP")
                res[1][col] = omega
                min_idx = idx
                break
            idx += 1
        col += 1
    save_data(res, "regression_smallest_omega_stability_simulation_finer.csv")

# Simulation that makes the GIFs
def gif_simulation():
    # Initialise the parameters
    t_start = 0
    t_end = 25.2
    h = 0.1
    A = 25
    omega = 5
    L = 1
    theta_0 = 0.01
    t = np.arange(t_start, t_end, h)

    pivot = [0, 2]
    # Solve the system
    sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h, theta_0))

    for i in range(0, len(t) - 1):
        # Update position pivot and point mass
        pivot[1] = sol[i + 1, 2]
        x = pivot[0] - L * np.sin(sol[i + 1, 0])
        y = pivot[1] - L * np.cos(sol[i + 1, 0])

        # Position pivot and point mass
        x_values = [pivot[0], x]
        y_values = [pivot[1], y]

        # Set up plot layout
        fig, ax = plt.subplots()
        plt.xlim(-(L + 2), L + 2)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.ylim(-(L + 2), L + 2)
        plt.axis('on')
        plt.gca().set_aspect('equal', adjustable='box')

        # Color last frame red if the pendulum has found an equilibrium
        color = "black"
        if i == len(t) - 2:
            is_eq, pos = is_equilibrium(sol)
            print(f"{is_eq, pos}")
            if is_eq:
                color = "red"

        # Plot the pendulum
        plt.plot(x_values, y_values, color=color)
        ax.add_patch(plt.Circle((x, y), 0.1, fill=True, color=color))
        rect_size = 0.2
        ax.add_patch(
            plt.Rectangle((pivot[0] - (rect_size / 2), pivot[1] - (rect_size / 2)), rect_size, rect_size, fill=True,
                          color=color))
        plt.title(fr'Iteration {i}')

        # Save the plots
        if i < 10:
            plt.savefig(os.path.join("images", f"image_00{i}.jpg"), bbox_inches='tight', dpi=150)
        elif i < 100:
            plt.savefig(os.path.join("images", f"image_0{i}.jpg"), bbox_inches='tight', dpi=150)
        else:
            plt.savefig(os.path.join("images", f"image_{i}.jpg"), bbox_inches='tight', dpi=150)
        plt.clf()
        fig.clf()

    make_gif()