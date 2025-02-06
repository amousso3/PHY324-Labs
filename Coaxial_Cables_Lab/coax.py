# Authors: Maforikan Amoussou, Jaden Cordeiro
# Date: 2/2/2025
# Description: Analysis of the Coaxial Cable Data

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.constants import c, mu_0, epsilon_0
from scipy.stats import chi2
import matplotlib.pyplot as plt

def red_chi_squared(x, y, model_y, params, uncertainties):
    predicted_y = model_y(x, *params)

    v = y.size - len(params)

    chi = ((y - predicted_y) / uncertainties)**2
    chi = chi.sum() / v
    chi_prob = 1-chi2.cdf(chi,v)
    return [chi, y-predicted_y, chi_prob]

def plot_residuals(x, y, xerr, yerr, model_name, x_axis, y_axis):
    plt.errorbar(x, y, yerr=yerr, xerr = xerr, color='purple',marker='o',markersize =4,ls='',lw=1,label="residuals")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()
    return None

def plot_data2(x, y, y_unc, pred_y, model_name, x_axis, y_axis):
    colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-',]
    colors_err = ['blue', 'green', 'red', 'cyan']
    plt.errorbar(x, y, yerr=y_unc,color=colors_err[0],marker='o',markersize =5,ls='',lw=1,label="Time Delay Data")
    plt.plot(x, pred_y, colors[2], label=model_name, linewidth=1)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()
    return None

def model_processing(model, x, y, y_unc, guesses):
    popt, pcov = curve_fit(model, x, y, sigma=y_unc, absolute_sigma=True, p0 = guesses)
    pstd = np.sqrt(np.diag(pcov))
    return popt, pstd

def model(x, m, b):
    return m*x + b

#SPEED OF PROPOGATION

Cable1 = [154.5, 150.3,153.2,154.7,157.6,155,154.2]
Cable1_Unc = [0.2,0.5,0.6,0.5,0.3,0.2,0.8]

Cable2 = [313.3, 313.5, 313, 316.2,312.7,313]
Cable2_Unc=[0.2,0.5,2,0.3,0.3,1]

Cable3 = [489.2, 489,490,479.9,496,490.8,493]
Cable3_Unc = [0.2,2,2,0.5,0.2,0.2,1]

Cable4 = [767,769.2,771.8,779,776.4,769.4,770]
Cable4_Unc = [0.2,0.6,0.4,1,0.6,0.3,0.2]

delta_t = np.array([154.2, 313.6, 489.7, 771.8])
delta_unc = np.array([0.2, 0.4, 0.4, 0.2])
l = np.array([15.09, 30.5, 48.36, 75.84]) #cable lengths
l_unc = np.array([0.02, 0.02, 0.02, 0.02])

speed = np.average(2*l/delta_t)
speed_unc = 0.25 * np.linalg.norm(2 * (2*l/delta_t)*np.sqrt((delta_unc/delta_t)**2+(l_unc/l)**2))

print(speed)

print(speed_unc)

popt, pstd = model_processing(model, 2*l, delta_t, delta_unc, None)
chi, residuals, chi_prob = red_chi_squared(2*l, delta_t, model, popt , delta_unc)

print(chi, chi_prob)
print(popt[0], pstd[0])
print(1000000000/popt[0], 1000000000*pstd[0]*popt[0]**(-2))
print(popt[1], pstd[1])

plot_data2(2*l, (10**6)*delta_t, delta_unc, (10**6)*model(2*l, *popt), 'Linear Model', 'Length Traversed (m)', 'Delay Time (nanoseconds)')
plot_residuals(2*l, residuals, None, delta_unc, '', 'Length Traversed (m)', 'Residuals (nanoseconds)')

print((1000000000*speed)/299792458)
print((1000000000*speed_unc/299792458))





#ATTENUATION

Vi = np.array([4.11, 4.18, 4.36, 4.37])
Vi_unc = np.array([0.02, 0.01, 0.02, 0.01])
Vr = np.array([3.84, 3.49, 3.34, 2.80])
Vr_unc = np.array([0.02, 0.02, 0.02, 0.02])


att = 20 * np.log10(Vr/Vi)
att_unc = 20 * np.abs((Vi/Vr)*(1/np.log(10))*(Vr/Vi) * np.sqrt((Vi_unc/Vi)**2+(Vr_unc/Vr)**2))

att_per_m = att/l
att_per_m_unc = att_per_m * np.sqrt((att_unc/att)**2 + (l_unc/l)**2)
print(att, att_unc)
print(att_per_m, att_per_m_unc)

print(0.25 * np.linalg.norm([0.004, 0.002, 0.001, 0.001], ord=2))
print(np.average([-0.039,-0.051,-0.048,-0.051]))
