# Authors: Maforikan Amoussou, Jaden Cordeiro
# Date: 1/30/2025
# Description: Analysis of the Transmission line data

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2
import matplotlib.pyplot as plt


def access_data(file_name):
    data = pd.read_csv(file_name, comment='#')
    return data.to_numpy(dtype=float)

def red_chi_squared(x, y, model_y, params, uncertainties):
    predicted_y = model_y(x, *params)

    v = y.size - len(params)

    chi = ((y - predicted_y) / uncertainties)**2
    chi = chi.sum() / v
    chi_prob = 1-chi2.cdf(chi,v)
    return [chi, y-predicted_y, chi_prob]

def model_processing(model, x, y, y_unc, guesses):
    popt, pcov = curve_fit(model, x, y, sigma=y_unc, absolute_sigma=True, p0 = guesses)
    pstd = np.sqrt(np.diag(pcov))
    return popt, pstd

def plot_data2(x, y, y_unc, pred_y, model_name, x_axis, y_axis):
    colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-',]
    colors_err = ['blue', 'green', 'red', 'cyan']
    plt.errorbar(x, y, yerr=y_unc,color=colors_err[0],marker='o',markersize =5,ls='',lw=1,label=model_name)
    plt.plot(x, pred_y, colors[2], label=model_name, linewidth=1)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()
    return None

def plot_residuals(x, y, xerr, yerr, model_name, x_axis, y_axis):
    plt.errorbar(x, y, yerr=yerr, xerr = xerr, color='purple',marker='o',markersize =4,ls='',lw=1,label="residuals")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()
    return None

def myGauss(x, A, mean, width, base):
    return A*np.exp(-(x-mean)**2/(2*width**2)) + base

def parabolic(x, a, b, c, d):
    return a*(x-d)**2+b*x+c

def sinusoid(x, a, phi, d, c):
    return a*np.sin(c*x-phi) + d


data = access_data('data/6V_3V.csv')
current, voltage = data[:,4], data[:,6]

current_unc = current * 0.001 + (50*10**(-12))

voltage_unc_1 = voltage[:48] * 0.00015 + (225*10**(-6))
voltage_unc_2 = voltage[48:477] * 0.0002 + (350*10**(-6))
voltage_unc_3 = voltage[477:] * 0.00015 + (5*10**(-3))
voltage_unc = np.hstack((voltage_unc_1, voltage_unc_2, voltage_unc_3))

lb, ub = (420, 550)

model = sinusoid
p0 = [0.0001, 17, 1]
popt, pstd = model_processing(model, voltage[lb:ub], current[lb:ub], current_unc[lb:ub], p0)
chi, residuals, chi_prob = red_chi_squared(voltage[lb:ub], current[lb:ub], model, popt , current_unc[lb:ub])

plot_data2(voltage[lb:ub], current[lb:ub], current_unc[lb:ub], model(voltage[lb:ub], *popt), 'Gauss', 'Voltage (V)', 'Current (A)')
plot_residuals(voltage[lb:ub], residuals, voltage_unc[lb:ub], current_unc[lb:ub], 'Gauss', 'Voltage (V)', 'Current (A)')

print(chi_prob, chi)
print(popt, pstd)


