# Authors: Maforikan Amoussou, Jaden Cordeiro
# Date: 1/30/2025
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

def data_access(data_name):
    data = access_data('data/{d}.csv'.format(d=data_name))
    current, voltage = data[:,4], data[:,6]

    current_unc = current * 0.001 + (50*10**(-12))

    voltage_unc_1 = voltage[:48] * 0.00015 + (225*10**(-6))
    voltage_unc_2 = voltage[48:477] * 0.0002 + (350*10**(-6))
    voltage_unc_3 = voltage[477:] * 0.00015 + (5*10**(-3))
    voltage_unc = np.hstack((voltage_unc_1, voltage_unc_2, voltage_unc_3))

    plt.errorbar(voltage, current, xerr= voltage_unc, yerr=current_unc, linewidth=1, marker='o',markersize =5, ls='', label=data_name)
    plt.xlabel("Accelerating Voltage")
    plt.ylabel("Current")

    return current, voltage, current_unc, voltage_unc

data_access("6V_3V")
data_access("6V_2V")
data_access("6V_2.25V")
data_access("6V_2.5V")
data_access("6V_2.75V")
data_access("6V_1.75V")
plt.legend()
plt.show()

data_access("5V_2.5V")
data_access("5V_3V")
data_access("5V_2V")
plt.legend()
plt.show()

data_access("5.5V_2.5V")
data_access("5.5V_3V")
data_access("5.5V_2V")
plt.legend()
plt.show()

data_name = "6V_2.75V"
current, voltage, current_unc, voltage_unc = data_access(data_name)
plt.show()

#Remove trough outliers
current = np.delete(current, [533, 321])
current_unc = np.delete(current_unc, [533, 321])
voltage = np.delete(voltage, [533, 321])
voltage_unc = np.delete(voltage_unc, [533, 321])

#Get Trough Index
troughs = find_peaks(-current, distance=100)[0][2:]

#Check Troughs
troughs_current = [current[i] for i in troughs]
troughs_voltage = [voltage[i] for i in troughs]
troughs_voltage_unc = [voltage_unc[i] for i in troughs]
#plt.plot(troughs_voltage, troughs_current)
#plt.show()

# Voltage Dips
excite_voltage = np.diff(troughs_voltage) # MULTIPLY BY e
excite_voltage_unc = np.zeros(len(troughs_voltage_unc))
for i in range(len(excite_voltage_unc)-1):
    v = np.sqrt(troughs_voltage_unc[i]**2+excite_voltage_unc[i+1])
    excite_voltage_unc[i] = v

average_excite = np.average(excite_voltage)
average_excite_unc = (1/len(excite_voltage_unc)) * np.linalg.norm(excite_voltage_unc)

print("Average Excitation: {v} +- {u}".format(v=average_excite, u=average_excite_unc) )

error = ((average_excite - 4.9)/4.9) * 100
print("Error from expected value: {x} %".format(x=error))


#Fit Ranges

sig = 4
A = 10**(-9)
p0_dict={0:[A,15,sig,0],1:[A,20,sig,0],2:[A,25,sig,0],3:[A,30,sig,0],4:[A,35,sig,0]}

for i in range(len(troughs)-1):
    lb, ub = (troughs[i], troughs[i+1])

    model = myGauss
    p0 = p0_dict[i]
    popt, pstd = model_processing(model, voltage[lb:ub], current[lb:ub], current_unc[lb:ub], p0)
    chi, residuals, chi_prob = red_chi_squared(voltage[lb:ub], current[lb:ub], model, popt , current_unc[lb:ub])

    plot_data2(voltage[lb:ub], current[lb:ub], current_unc[lb:ub], model(voltage[lb:ub], *popt), 'Gauss', 'Accelerating Voltage (V)', 'Current (A)')
    plot_residuals(voltage[lb:ub], residuals, voltage_unc[lb:ub], current_unc[lb:ub], 'Gauss', 'Accelerating Voltage (V)', 'Current (A)')

    print("Chi-Probability: {x:.1f}".format(x=chi_prob),"Reduced Chi Squared: {x:.1f}".format(x=chi))
    print(popt, pstd)


