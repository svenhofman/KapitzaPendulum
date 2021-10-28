import numpy as np
import matplotlib.pyplot as plt
import os
from runge_kutta_4 import aux_runge_kutta_4
from runge_kutta_4 import Parameters
from numpy import genfromtxt
import matplotlib.mlab as mlab


def load_data(name):
    return genfromtxt(os.path.join("data", name), delimiter=",")


# Figure 4
def order_of_convergence():
    A = 40
    omega = 8
    t_end = 10
    n_steps = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    h_arr = t_end / n_steps

    theta = []
    y = []

    for idx in range(0, len(n_steps)):
        sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h_arr[idx]))
        theta.append(sol[int(n_steps[idx] / 2), 0])
        y.append(sol[int(n_steps[idx] / 2), 2])

    theta_conv = []
    y_conv = []

    for k in range(0, len(n_steps) - 3):
        theta_conv.append(np.log2(abs(theta[k] - theta[k + 1]) / abs(theta[k + 1] - theta[k + 2])))
        y_conv.append(np.log2(abs((y[k]) - y[k + 1]) / abs(y[k + 1] - y[k + 2])))

    plt.figure(1)
    plt.title(r'Order of convergence of angle $\theta$')
    plt.ylabel(r'$q$')
    plt.xlabel(r'$h$')
    xP = [r'$1/2^8$', r'$1/2^9$', r'$1/2^{10}$', r'$1/2^{11}$',
          r'$1/2^{12}$', r'$1/2^{13}$']
    plt.xticks(list(range(len(x_values))), x_values)
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.plot(theta_conv, marker='o', color='k')
    plt.savefig(os.path.join('images', 'theta-convergence.pdf'))

    plt.figure(2)
    plt.title(r'Order of convergence of $y_p$')
    plt.xlabel(r'$h$')
    plt.ylabel(r'$q$')
    x_values = [r'$1/2^8$', r'$1/2^9$', r'$1/2^{10}$', r'$1/2^{11}$',
                r'$1/2^{12}$', r'$1/2^{13}$']
    plt.xticks(list(range(len(x_values))), x_values)
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.plot(y_conv, marker='o', color='k')
    plt.savefig(os.path.join('images', 'yp-convergence.pdf'))
    plt.show()


# Figure 5
def no_oscillation():
    A = 0
    omega = 5
    t_end = 30
    h = 0.01

    sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h))
    t = np.arange(0, t_end + h, h)

    plt.plot(t, sol[:, 0], label=r'$\theta$', color='k')
    plt.plot(t, sol[:, 1], '--k', label=r'$\dot{\theta}$')
    plt.xlabel('t (seconds)')
    plt.title('Angle and angular velocity over time with no oscillation')
    plt.legend()

    plt.savefig(os.path.join('images', 'no_oscillation.pdf'))
    plt.show()


# Figure 6
def oscillation_stability():
    A = 40
    omega = 8
    t_end = 30
    h = 0.01

    sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h))
    t = np.arange(0, t_end + h, h)

    dummy, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(t, sol[:, 0], 'k')
    angle_deflection = sol[:, 0] - np.pi
    angle_deflection = angle_deflection / (2 * np.pi) * 360
    ax2.plot(t, angle_deflection, 'k', alpha=0.3)

    ax1.set_xlabel('t (seconds)')
    ax1.set_ylabel(r'$\theta$')
    ax2.set_ylabel(r'angular deflection (degrees)')
    plt.title('Angle over time with vertical oscillation')

    plt.savefig(os.path.join('images', 'oscillation_stability.pdf'))
    plt.show()


# Figure 7
def bottom_unstable():
    A = 25
    omega = 5
    t_end = 30
    h = 0.01
    theta_0 = 0.01

    sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h, theta_0))
    t = np.arange(0, t_end + h, h)

    plt.plot(t, sol[:, 0], color='k')
    plt.xlabel('t (seconds)')
    plt.ylabel(r'$\theta$')
    plt.title('Angle over time with vertical oscillation, starting near bottom eq.')

    plt.savefig(os.path.join('images', 'bottom_unstable.pdf'))
    plt.show()


# Figure 8
def top_unstable():
    A = 41
    omega = 9
    t_end = 30
    h = 0.01

    sol = aux_runge_kutta_4(Parameters(A, omega, t_end, h))
    t = np.arange(0, t_end + h, h)

    plt.plot(t, sol[:, 0], color='k')
    plt.xlabel('t (seconds)')
    plt.ylabel(r'$\theta$')
    plt.title('Angle over time with vertical oscillation')

    plt.savefig(os.path.join('images', 'top_unstable.pdf'))
    plt.show()


# Figure 9
def smallest_omega_stability():
    res = load_data("smallest_omega_stability_simulation.csv")
    plt.plot(res[0, :], res[1, :], 'k,')
    plt.xlabel('$A$')
    plt.ylabel('$\omega$')
    plt.title('Angular frequency plotted against amplitude')

    plt.savefig(os.path.join('images', 'smallest_omega_stability.pdf'))
    plt.show()


# Figure 10
def smallest_omega_regression_line():
    res = load_data("regression_smallest_omega_stability_simulation_finer.csv")
    plt.plot(res[0, :], res[1, :], 'k--', label="Stability line")
    slope, intercept = np.polyfit(res[0, :], res[1, :], 1)
    print(f"Slope of regression line is approximately {round(slope, 2)}")
    plt.plot(res[0, :], slope * res[0, :] + intercept, 'k', label="Regression line")

    plt.xlabel('$A$')
    plt.ylabel('$\omega$')
    plt.title('Angular frequency plotted against amplitude')
    plt.legend()

    plt.savefig(os.path.join('images', 'smallest_omega_stability_regression.pdf'))
    plt.show()


# Figure 11
def plot_grid_amplitude_omega():
    # Grid for A from 1 to 300, 150 values
    A_range = np.linspace(1, 300, 150)
    # Grid for omega from 1 to 40, 50 values
    omega_range = np.linspace(1, 40, 50)

    # Data is generated by calling the simulation with the configuration given in the title
    data = load_data("A_range[1,300](150)_omega_range[1,40](50)_h=0.01_t_end=50.csv")

    # Plot line from literature Kapitza
    g, L = 9.81, 1
    slope1 = 1 / np.sqrt(2 * g * L)
    idx = next(x[0] for x in enumerate(A_range) if x[1] > 178)
    line1 = slope1 * A_range[:idx]

    # Find regression line
    first_stable_arr = np.zeros((2, 150))
    row, col = 0, 0
    for A in np.linspace(1, 300, 150):
        for omega in np.linspace(1, 40, 50):
            if data[row, col] == 1:
                first_stable_arr[0, col] = A
                first_stable_arr[1, col] = omega
                break
            row += 1
        row = 0
        col += 1

    start_idx = 70
    slope2, intercept2 = np.polyfit(first_stable_arr[0, start_idx:], first_stable_arr[1, start_idx:], 1)
    line2 = slope2 * A_range + intercept2

    x_values_neg = []
    y_values_neg = []
    x_values_zero = []
    y_values_zero = []
    x_values_one = []
    y_values_one = []

    for i in range(len(omega_range)):
        for j in range(len(A_range)):
            if int(data[i, j]) == -1:
                x_values_neg.append(A_range[j])
                y_values_neg.append(omega_range[i])
            elif int(data[i, j]) == 0:
                x_values_zero.append(A_range[j])
                y_values_zero.append(omega_range[i])
            elif int(data[i, j]) == 1:
                x_values_one.append(A_range[j])
                y_values_one.append(omega_range[i])
            else:
                print(f"UNKNOWN: {int(data[i, j])}")

    marker_size = 2
    plt.plot(A_range[:idx], line1, color="black", label=f"$\omega_1 = {round(slope1, 2)}A$")
    plt.plot(A_range, line2, "-.", color="black", label=f"$\omega_2 = {round(slope2, 2)}A + {round(intercept2, 2)}$")
    plt.scatter(x_values_neg, y_values_neg, color="red", s=marker_size, label="No stability")
    plt.scatter(x_values_zero, y_values_zero, color="green", s=marker_size, label="Bottom equilibrium")
    plt.scatter(x_values_one, y_values_one, color="blue", s=marker_size, label="Top equilibrium")
    plt.xlabel("$A$")
    plt.ylabel("$\omega$")
    plt.title("Angular frequency plotted against amplitude")
    plt.legend()
    plt.savefig(os.path.join('images', "amplitude-omega.pdf"))

    plt.show()


# Figure 12
def plot_grid_damping_angle0():
    angle_range = np.linspace(0, 2 * np.pi, 150)
    damping_range = np.linspace(0, 1.00, 100)

    data = load_data("data-demp.csv")

    x_values_neg = []
    y_values_neg = []
    x_values_zero = []
    y_values_zero = []
    x_values_one = []
    y_values_one = []

    for i in range(len(damping_range)):
        for j in range(len(angle_range)):
            if int(data[i, j]) == -1:
                x_values_neg.append(damping_range[i])
                y_values_neg.append(angle_range[j])
            elif int(data[i, j]) == 0:
                x_values_zero.append(damping_range[i])
                y_values_zero.append(angle_range[j])
            elif int(data[i, j]) == 1:
                x_values_one.append(damping_range[i])
                y_values_one.append(angle_range[j])
            else:
                print(f"UNKNOWN: {int(data[i, j])}")

    marker_size = 2
    plt.scatter(x_values_neg, y_values_neg, color="red", s=marker_size, label="No stability")
    plt.scatter(x_values_zero, y_values_zero, color="green", s=marker_size, label="Bottom equilibrium")
    plt.scatter(x_values_one, y_values_one, color="blue", s=marker_size, label="Top equilibrium")
    plt.xlabel(r"$b_m$")
    plt.ylabel(r"$\theta_0$")
    plt.yticks(np.array([0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3, 2 * np.pi]),
               [r'$0$', r'$\frac{\pi}{3}$', r'$\frac{2\pi}{3}$', r'$\pi$', r'$\frac{4\pi}{3}$',
                r'$\frac{5\pi}{3}$', r'$2\pi$'])
    plt.title(r"Initial angular position plotted against damping coefficient")
    plt.legend()
    plt.savefig(os.path.join('images', "theta-bp.pdf"))

    plt.show()


plot_grid_amplitude_omega()
