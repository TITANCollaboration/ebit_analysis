"""
Program to analyse RFA spectra.
First step involves filtering data for background spectrum.
Second step involves fitting all peaks or summing peak areas.
Third step involves fitting Voltage vs Count curve with error function.

Fitting program starts here
"""

# Import all necessary python headers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy.signal import find_peaks
import itertools
from scipy.special import erf

# Define some functions to be used
# Simple Gaussian distribution with amplitude, mean and width
def Gaus(x_local, amplitude, mean, width):
    return amplitude * np.exp(-((x_local - mean) / width) ** 2)


# Fitting function for peaks based on a rough Centroid guess for the peak. This method is useful in creating automated
# script to couple with peak finding algorithms. Centroid is rough peak in microseconds which is changed to channel
# numbers inside the function. This function requires following inputs:
# x variables; y variables; lower_cut (us); upper cut (us); width of gauss peak; and sampling dwell time used
# in DAQ recording. Dwell time was in nanoseconds but here you need to enter in microseconds, e.g. 30 ns = 0.03 us.

def Err(x_local, amplitude, mean, width, epsilon):
    return 0.5*amplitude * erf(-((x_local - mean) / (np.sqrt(2)*width)) ) + epsilon


def FitFuncError(x_local_for_fit, y_local_for_fit, Cent, width, epsilon):
    # This fitting function utilizes optimize function from scipy. It requires a function, x and y variables, and
    # initial parameters. If initial parameters are way off, fit will not converge.
    init_val = [max(y_local_for_fit), Cent, width, epsilon]  # Initial parameters for fitting
    best_val, covar = optimize.curve_fit(Err, x_local_for_fit, y_local_for_fit, init_val)
    return best_val, covar


def SumPeak(c, li, hi, dwell_time):
    rl_local = int(li / dwell_time)  # range converted to integer count
    rh_local = int(hi / dwell_time)  # range converted to integer count
    x_temp = c[rl_local:rh_local]  # x range extracted from full range for fitting
    return sum(x_temp)


# ----------------------------------------
# Input variables for the program
# Figure flags
dpiCount = 70  # A variable to control size of the figure.
Flag_Fig1 = 0  # Flag for plotting/displaying figure
colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])  # Colors database for plotting

# Fitting parameters
time_const = 0.03  # Bin width during measurement in microseconds (original 30 ns)
wid = 5  # Initial variables for fitting peak - width of peak (in us)

# File read and data sorting performed here. All data exists in single file.
# Data is row wise and not column wise. Therefore, we need to transpose matrix.
# First spectrum is background
# Second spectrum is Data.
# Need to subtract these spectra to calculate final spectrum for analysis.

df = pd.read_csv("./data/Rb_Rfa_chargebred_stepsize5V_spectra_29_01_2020__1_54_38_PM.txt", delimiter='\t',
                 skiprows=6)  # , header=0, usecols=[np.arange(5, 40, 1)])
# df = pd.read_csv("./data/Rb_Rfa_reflected_stepsize5V_spectra_30_01_2020__2_26_52_PM.txt", delimiter='\t',
#                  skiprows=6)  # , header=0, usecols=[np.arange(5, 40, 1)])


df = df.T   # Transpose dataframe
data = df[6:]   # Use only selected spectrum.
time = pd.Series(time_const * np.arange(len(data)))  # Creating an x-axis (time) as this is a 1D data

plt.figure(0, figsize=plt.figaspect(0.5), dpi=200)
plt.plot(time, data[21]-data[20], '-k', label='Data')
plt.legend(loc="upper right", fontsize=14)
plt.xlim(15, 55)
plt.xticks(fontsize=14)
plt.xlabel('RFA Voltage (V)', fontsize=18)
plt.ylim(-2, 20)
plt.yticks(fontsize=14)
plt.ylabel('Counts', fontsize=18)
plt.savefig('Data.png')  # FIle saving


# Array for getting sum of peak area
PeakCount = []
for i in np.arange(0, data.shape[1] / 2, 1):
    s = (data[2 * i + 1] - data[2 * i]).reset_index()
    ssum = SumPeak(s[0], 20, 55, time_const)
    PeakCount.append(ssum)

# Array for getting Retarding Voltage used in the spectra
RF_Vol = []
for i in np.arange(0, data.shape[1] / 2, 1):
    RF_Vol.append(df.loc[df.index[4]][2 * i])

# Error function fit is performed here
fit_par, var = FitFuncError(RF_Vol, PeakCount, 2000, wid, 500)  # fit_par are fit parameters

# Figure plotting of RFA data
plt.figure(1, figsize=plt.figaspect(0.5), dpi=200)
plt.plot(RF_Vol, PeakCount, 'bo')
plt.plot(RF_Vol, Err(RF_Vol, fit_par[0], fit_par[1], fit_par[2], fit_par[3]), color=next(colors), label='Error Func Fit')
plt.plot(RF_Vol, Gaus(RF_Vol, fit_par[0], fit_par[1], fit_par[2]), color=next(colors), label='Gauss')
plt.legend(loc="upper right", fontsize=14)
# plt.xlim(15, 75)
plt.xticks(fontsize=14)
plt.xlabel('RFA Voltage (V)', fontsize=18)
# plt.ylim(0, round(max(y_raw), -2))
plt.yticks(fontsize=14)
plt.ylabel('Counts', fontsize=18)
plt.savefig('Fit_RFA.png')  # FIle saving

print(fit_par)
plt.show()
