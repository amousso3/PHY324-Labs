# Authors: Maforikan Amoussou, Jaden Cordeiro
# Date: 1/30/2025
# Description: Analysis of the Transmission line data

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.constants import c, mu_0, epsilon_0
import matplotlib.pyplot as plt 

def access_data(file_name):
    data = pd.read_csv(file_name, comment='#')
    return data.to_numpy(dtype=float)

def red_chi_squared(x, y, model_y, params, uncertainties):
    predicted_y = model_y(x, *params)
    
    v = y.size - len(params)

    chi = ((y - predicted_y) / uncertainties)**2
    chi = chi.sum() / v
    return [chi, y-predicted_y]

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

def lc_model(x, m, b):
    return m*x + b

def attenuation(Vo, Vi):
    return 20*np.log10(Vo/Vi)

#def unc_attenuation()

data = access_data('data/PC_Ex1_Data.csv')

lc_unit, delta_t, delta_unc = data[:,0], data[:,1], data[:,4]

popt, pstd = model_processing(lc_model, lc_unit, delta_t, delta_unc, None)

chi, residuals = red_chi_squared(lc_unit, delta_t, lc_model, popt , delta_unc)

print(chi)
print(popt[0])
plot_data2(lc_unit, delta_t, delta_unc, lc_model(lc_unit, *popt), 'LC Slope', 'LC Units', 'Delay Time')
plot_residuals(lc_unit, residuals, None, delta_unc, 'LC Slope', 'LC units', 'Residuals')
