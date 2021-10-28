import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

from runge_kutta_4 import runge_kutta_4

OTHER = -1
BOTTOM = 0
TOP = 1


# Auxiliary function that is calls runge_kutta_4() for all time steps
def aux_runge_kutta_4(param):
    t = np.arange(0, param.t_end + param.h, param.h)
    sol = np.empty((len(t), 4))

    sol[0] = [param.theta_0, 0.0, 0.0, -param.A / param.omega]
    for i in range(0, len(t) - 1):
        sol[i + 1] = runge_kutta_4(sol[i], t[i], param)
    return sol


# Checks for stability with the criterion defined in the theory
def is_equilibrium(sol, err=0.1):
    # Top equilibrium
    if abs(((sol[len(sol) - 1, 0] + 1) % (2 * np.pi) - 1) - np.pi) < err and abs(sol[len(sol) - 1, 1]) < err:
        return True, TOP

    # Bottom equilibrium
    if abs((sol[len(sol) - 1, 0] + 1) % (2 * np.pi) - 1) < err and abs(sol[len(sol) - 1, 1]) < err:
        return True, BOTTOM

    return False, OTHER


# Makes a gif from all the .jpg files and removes the .jpg files afterwards
def make_gif():
    jpg_dir = os.path.join(".", "images")
    images = []
    for file_name in sorted(os.listdir(jpg_dir)):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(jpg_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(os.path.join("images", 'simulation.gif'), images)

    for file in os.listdir(jpg_dir):
        if file.endswith('.jpg'):
            os.remove(os.path.join(jpg_dir, file))
