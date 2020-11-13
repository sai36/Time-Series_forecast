from __future__ import division
from sys import exit
from math import sqrt
from numpy import array
from scipy.optimize import fmin_l_bfgs_b
import csv
import pandas as pd
import psycopg2
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


def damped_multiplicative(x, m, fc, alpha = None, beta = None, gamma = None, phi = None):

    Y = x[:]

    if (alpha == None or beta == None or gamma == None):
        initial_values = array([0.3, 0.1, 0.1, 0.1])
        boundaries = [(0, 1), (0, 1), (0, 1), (0, 1)]
        type = 'damped_multiplicative'

        parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
        alpha, beta, gamma, phi = parameters[0]

    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / float(m ** 2)]
    s = [Y[i] / a[0] for i in range(m)]
    y = [(a[0] + b[0]) * s[0]]
    rmse = 0

    for i in range(len(Y) + fc):

        if i == len(Y):
            Y.append(a[-1] * (b[-1] ** phi) * s[-m])

        a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] * (b[i]) ** phi))
        b.append(beta * (a[i + 1] /a[i]) + (1 - beta) * (b[i]) ** phi)
        s.append(gamma * (Y[i] / (a[i] *(b[i]) ** phi)) + (1 - gamma) * s[i])
        y.append(a[i + 1] * (b[i + 1] ** phi ) * s[i + 1])

    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
