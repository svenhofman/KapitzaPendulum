import numpy as np


# A class that initialises all parameters
class Parameters:

    def __init__(self, A, omega, t_end, h, theta_0=np.pi - 0.01, b_m=0.5, g=9.81, L=1, m=1):
        self.A = A
        self.omega = omega
        self.theta_0 = theta_0
        self.t_end = t_end
        self.h = h
        self.g = g
        self.b_m = b_m
        self.L = L
        self.m = m


# The vector function that represents the system of ODES 
def F(prev, t, param):
    # prev = [theta, phi, y_p, v_p]
    theta = prev[1]
    phi = -param.A * np.sin(param.omega * t) * np.sin(prev[0]) / param.L - param.b_m * prev[
        1] / param.m - param.g * np.sin(prev[0]) / param.L
    y_p = prev[3]
    v_p = param.A * np.sin(param.omega * t)
    return np.array([theta, phi, y_p, v_p])


# The Runge-Kutta 4 method
def runge_kutta_4(prev, t, param):
    k1 = param.h * F(prev, t, param)
    k2 = param.h * F(prev + 0.5 * k1, t + 0.5 * param.h, param)
    k3 = param.h * F(prev + 0.5 * k2, t + 0.5 * param.h, param)
    k4 = param.h * F(prev + k3, t + param.h, param)
    return prev + (k1 + 2 * (k2 + k3) + k4) / 6.0
