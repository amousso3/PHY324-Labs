# Authors: Maforikan Amoussou, Jaden Cordeiro
# Date: 3/6/2025
# Description: Analysis of the Transmission line data

import numpy as np
import pandas as pd
import scipy.constants
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import rc

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size': 20}
rc('font', **font)

e = scipy.constants.e

def access_data(file_name):
    data = pd.read_csv(file_name, comment='#')
    return data.to_numpy(dtype=float)

def red_chi_squared(x, y, model_y, params, uncertainties):
    predicted_y = model_y(x, *params)

    v = y.size - len(params)

    chi_squared = np.sum(((y - predicted_y) / uncertainties)**2)
    chi_prob = 1-chi2.cdf(chi_squared,v)
    chi_red = chi_squared/v
    return [chi_red, y-predicted_y, chi_prob]

def model_processing(model, x, y, y_unc, guesses):
    popt, pcov = curve_fit(model, x, y, sigma=y_unc, absolute_sigma=True, p0=guesses)
    pstd = np.sqrt(np.diag(pcov))
    return popt, pstd

def plot_data2(x, y,x_unc, y_unc, pred_y, model_name, x_axis, y_axis):
    colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-',]
    colors_err = ['blue', 'green', 'red', 'cyan']
    plt.errorbar(x, y, xerr=x_unc, yerr=y_unc,color=colors_err[0],marker='o',markersize =5,ls='',lw=1,label="Current Data")
    plt.plot(x, pred_y, colors[2], label=model_name, linewidth=1)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()
    return None

def plot_residuals(x, y, xerr, yerr, x_axis, y_axis):
    plt.errorbar(x, y, yerr=yerr, xerr = xerr, color='purple',marker='o',markersize =4,ls='',lw=1,label="Residuals")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()
    return None

def model(x, a, c):
    return a*x+c

data = access_data("Wavelength.csv")
tick, fringes = data[:,0]*10**(-6), data[:,1]
tick_unc = np.ones_like(tick) * 10**(-6)
fringes_unc = np.ones_like(fringes)

p0 = [0.5, 0]
popt, pstd = model_processing(model, tick, fringes, fringes_unc, p0)
red_chi, residuals, chi_prob = red_chi_squared(tick, fringes, model, popt, fringes_unc)
plot_data2(tick, fringes, tick_unc, fringes_unc, model(tick, *popt),"Linear Fit", "Distance Moved [m]", "Counts of Fringes Appeared")
plot_residuals(tick, residuals, tick_unc, fringes_unc, "Distance Moved [m]", "Counts of Fringes Appeared")
print(red_chi, chi_prob)
wavelength = 2 * (popt[0])**-1
wavelength_unc = (-1) * 2 * popt[0]**(-2) * pstd[0]
print(wavelength, wavelength_unc)

p0 = [2, 0]
popt, pstd = model_processing(model, fringes, tick, tick_unc, p0)
red_chi, residuals, chi_prob = red_chi_squared(fringes, tick, model, popt, tick_unc)
plot_data2(fringes, tick, fringes_unc, tick_unc, model(fringes, *popt),"Linear Fit", "Counts of Fringes Appeared", "Distance Moved [m]")
plot_residuals(fringes, residuals, fringes_unc, tick_unc, "Counts of Fringes Appeared", "Distance Moved [m]")
print(red_chi, chi_prob)
wavelength = 2 * (popt[0])
wavelength_unc = pstd[0] * 2
print(wavelength, wavelength_unc)


