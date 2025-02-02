# Authors: Maforikan Amoussou, Jaden Cordeiro
# Date: 1/30/2025
# Description: Analysis of the Transmission line data

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.constants import c, mu_0, epsilon_0
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

def lc_model(x, m, b):
    return m*x + b

def attenuation(Vr, Vi):
    return 20*np.log10(Vr/Vi)

def unc_attenuation(Vr, uVr, Vi, uVi):
    return np.sqrt(((20/(Vi*np.log(10)))*uVi)**2+((20/(Vr*np.log(10)))*uVr)**2)

data = access_data('data/PC_Ex1_Data.csv')

lc_unit, delta_t, delta_unc = data[:,0], (10**-6)*data[:,1], (10**-6)*data[:,4]

Vi = np.array([4.11,4.18,4.36,4.37])
uVi = np.array([0.02, 0.01, 0.02, 0.01])
Vr = np.array([3.84,3.49,3.34,2.80])
uVr = np.array([0.02, 0.02, 0.02, 0.02])

atten = attenuation(Vr, Vi)
uatten = unc_attenuation(Vr, uVr, Vi, uVi)
print(np.round(atten,2), np.round(uatten,2))


popt, pstd = model_processing(lc_model, lc_unit, delta_t, delta_unc, None)
chi, residuals, chi_prob = red_chi_squared(lc_unit, delta_t, lc_model, popt , delta_unc)

plot_data2(lc_unit, (10**6)*delta_t, delta_unc, (10**6)*lc_model(lc_unit, *popt), 'LC Slope', 'LC Units', 'Delay Time (microseconds)')
plot_residuals(lc_unit, residuals, None, delta_unc, 'LC Slope', 'LC units', 'Residuals (microseconds)')

print(np.round(chi_prob, 1))
print((1/popt[0]), (np.sqrt((pstd[0]*(popt[0]**-2))**2)))
print(np.round((10**6)*popt[0], 3), np.round((10**6)*pstd[0], 3))
print(np.round((10**6)*popt[1], 1), np.round((10**6)*pstd[1], 1))
inductance = 1.5 * 10**-3
capacitance = 0.01 * 10**-6

print(np.sqrt(1/(inductance*capacitance)))

unc_LC = np.array([0.15*10**-3, 0.00003*10**-6])
lc = np.array([-0.5*capacitance*(inductance*capacitance)**-1.5, -0.5*inductance*(inductance*capacitance)**-1.5])

print(np.sqrt(np.dot(unc_LC**2, lc**2)))
