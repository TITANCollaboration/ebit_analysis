""                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        """
Program to calculate charge states for EBIT data. First step involves fitting all peaks. Second step is getting
charge state ratios from peak centroids.
Fitting program starts here
"""

# Import all necessary python headers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy.signal import find_peaks
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy import optimize
from scipy.signal import argrelextrema
import itertools

# Define some functions to be used
# Simple Gaussian distribution with amplitude, mean and width
def Gaus(x_local, amplitude, mean, width):
    return amplitude * np.exp(-((x_local - mean) / width) ** 2)


# Fitting function for peaks based on a rough Centroid guess for the peak. This method is useful in creating automated
# script to couple with peak finding algorithms. Centroid is rough peak in microseconds which is changed to channel
# numbers inside the function. This function requires following inputs:
# x variables; y variables; lower_cut (us); upper cut (us); width of gauss peak; and sampling dwell time used
# in DAQ recording. Dwell time was in nanoseconds but here you need to enter in microseconds, e.g. 30 ns = 0.03 us.
def FitFuncCentroid(x_local_for_fit, y_local_for_fit, Cent, width, dwell_time):
    global best_val, covar
    range_cut = 2.0 * width
    # Centroid+-range_cut gives the left and right range for fitting.
    rl_local = int((Cent - range_cut) / dwell_time)  # range converted to integer count
    rh_local = int((Cent + range_cut) / dwell_time)  # range converted to integer count
    init_val = [max(y_local_for_fit), Cent, width]  # Initial parameters for fitting

    x_temp = x_local_for_fit[rl_local:rh_local]  # x range extracted from full range for fitting
    y_temp = y_local_for_fit[rl_local:rh_local]  # y range extracted from full range for fitting

    # This fitting function utilizes optimize function from scipy. It requires a function, x and y variables, and
    # initial parameters. If initial parameters are way off, fit will not converge.

    try:
            best_val, covar = optimize.curve_fit(Gaus, x_temp, y_temp, init_val)
    except RuntimeError:
            pass
    # Errors in the fit parameters can be calculated below but I will calculate them later in-line to get more control.
    # errors = np.sqrt(np.diag(covar))
    # print('Amplitude is ', best_val[0], '(', errors[0], ')')
    # print('Mean is ', best_val[1], '(', errors[1], ')')
    # print('Width is ', best_val[2], '(', errors[2], ')')
    return best_val, covar


# Function for generating a reverse ordered array. nl is lower limit and nh is the upper limit.
# It generates array from nl to nh including both numbers.
def reverse_count(nl, nh):
    init_arr = []
    count = nh
    for i in range(nl, nh+1):
        init_arr.append(count)
        count = count - 1
    return init_arr


# Rounding of an array elements to nth decimal
def round_array(arr, n):
    round_arr = []
    for i in range(len(arr)):
        b = round(arr[i], n)
        round_arr.append(b)
    return round_arr


# Normalized charge state for comparison
def norm_charge(charge_arr, index):
    norm_arr = []
    for i in range(len(charge_arr)):
        b = charge_arr[index] / charge_arr[i]
        norm_arr.append(b)
    return norm_arr


# ----------------------------------------
# Input variables for the program
# Figure flags
dpiCount = 70  # A variable to control size of the figure.
Flag_Fig0 = 0  # Flag for plotting/displaying figure - Raw spectra
Flag_Fig1 = 1  # Flag for plotting/displaying figure - Subtracted spectra and fits
Flag_Fig2 = 0  # Flag for plotting/displaying figure - Width vs Mean
Flag_Fig3 = 1  # Flag for plotting/displaying figure - Plots of Time ratio and Charge ratio
Flag_Fig4 = 1  # Flag for plotting/displaying final figure - Peaks identified by charge state

fig = plt.figure()
ax = fig.add_subplot(111)

colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])  # Colors database for plotting

# Fitting parameters
time_const = 0.03  # Bin width during measurement in microseconds (Default 30 ns)
width = .3  # Initial variables for fitting peak - width of peak (in us)
dis = 40  # Distance (in channel) between two consecutive peak search
h = 75  # Threshold for peak search
mass_Cs = 133  # Mass of ion
z_Cs = 55  # Atomic number of ion
length = 1500  # Length of spectrum used in analysis (Max channel number)
length_min = 1
# ----------------------------------------
# An inline test function for Gaussian distribution, may be useless for this program but good tool for future use.
# gaus = lambda x, *p: p[0]*exp(-((x-p[1])/p[2])**2)

# File read and data sorting performed here. Zeroth (First) line is header. usecols could be set to 1 but I have left
# it as 'range' to make it future proof. Range selects number of columns to be imported.
# data_file is the data file and data_file_bkg is the background file.
#data_file = np.loadtxt('Cs.txt', dtype=int, comments='#', delimiter=None, skiprows=1, unpack=False)
#data_file_bkg = np.loadtxt('Cs1.txt', dtype=int, comments='#', delimiter=None, skiprows=1, unpack=False)
data = input("Enter nonbkg ")
bkg = input("Enter bkg ")
input1 ='./data/{0}.txt'.format(data)
input2 ='./data/{0}.txt'.format(bkg)
data_file = pd.read_csv(input1, header=0, usecols=[i for i in range(1)])
data_file_bkg = pd.read_csv(input2, header=0,
                            usecols=[i for i in range(1)])
x = pd.Series(time_const * np.arange(len(data_file)))  # Creating an x-axis (time) as this is a 1D data
dd = pd.concat([x[length_min:length], data_file[length_min:length]], axis=1)  # Concatenating data in a pandas dataframe from x and y variables.
dd.columns = ['Time', 'Counts']  # Adding column headers manually
# x = dd.loc[:, 'Time']
y_raw = dd.loc[:, 'Counts']  # y raw data

dd = pd.concat([x[length_min:length], data_file_bkg[length_min:length]], axis=1)  # Creating datframe for background data
dd.columns = ['Time', 'Counts']  # Adding column headers manually
y_back = dd.loc[:, 'Counts']  # y background data

# Subtracting background and creating the spectrum for analysis
x_vals = dd.loc[:, 'Time']  # Final x values
y_vals = y_raw - y_back  # Final subtracted y values

if Flag_Fig0 == 1:
    # Section to see visually how spectra look.
    plt.figure(0, figsize=plt.figaspect(0.5), dpi=dpiCount)
    plt.plot(x_vals, y_raw, '-g', Linewidth=1, label="Raw Data")
    plt.plot(x_vals, y_back, '-r', Linewidth=1, label="Background")
    plt.plot(x_vals, y_vals, '-k', Linewidth=1, label="Subtracted Data")
    plt.legend(loc="upper right", fontsize=18)
    plt.xlim(10, 70)
    plt.xticks(fontsize=14)
    plt.xlabel(r'Time ($\mu$ s)', fontsize=18)
    plt.ylim(0, round(max(y_raw), -2))
    plt.yticks(fontsize=14)
    plt.ylabel('Counts', fontsize=18)
    # plt.savefig('Raw.png')# FIle saving

if Flag_Fig1 == 1:
    # Plot final subtracted data. The same graph will be used for showing fitted data.
    plt.figure(1, figsize=plt.figaspect(0.5), dpi=dpiCount)
    plt.plot(x_vals, y_vals, '-k', Linewidth=1, label="Data")

# Fitting of several peaks will start here

# Finding peaks in the processed spectrum
#peaks, _ = find_peaks(y_vals, distance=dis, height=h)
ysmooth = signal.savgol_filter(y_vals, 25, 3)
maxInd = argrelextrema(ysmooth, np.greater, order=10)
smpeaks = ysmooth[maxInd[0]]
#print(smpeaks)
negMax = np.where(smpeaks <1)
negMax1 = negMax[0].tolist()
maxInd1=maxInd[0].tolist()
smpeaks1 = smpeaks.tolist()
for i in sorted(negMax1, reverse = True):
    del maxInd1[i]
    del smpeaks1[i]
#print(smpeaks1)
smpeaks3=np.asarray(smpeaks1)
maxPk = np.amax(smpeaks)
noise = np.where(smpeaks3 < 0.01*maxPk)
noise1 = noise[0].tolist()
#print(maxPk)
for i in sorted(noise1, reverse = True):
    del maxInd1[i]
    del smpeaks1[i]
peaks1= [0.03 * i for i in maxInd1]
#print(peaks1)
peaks2 = np.asarray(peaks1)
peaks = np.asarray(maxInd1)
#print(peaks)
#print(peaks2)
smpeaks2=np.asarray(smpeaks1)
#print(smpeaks2)

plt.plot(peaks2 * time_const, y_vals[peaks2], "x") # Plotting peak with x markers
plt.plot(peaks2, smpeaks2,'x',color='red',markersize=6)
for xy in zip(peaks2,smpeaks2):
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
#plt.show()
# Create an empty array for storing calculated parameters. len(peak) makes the length of array based on number
# of peaks found, and 6 column is twice the fitted number of parameter. If fit function has more parameters, change this
# number accordingly.
Peak_parameters = np.zeros((len(peaks), 6))
# Loop starts for fitting and storing information in array
for npeak in range(len(peaks)):  # First level for rows
    i = npeak + 0  # Diagnostic step to change index of peak. In some cases it can find abnormal peaks.
    Centroid_guess = time_const * peaks[i]  # Initial guess for Centroid. It can be a number or in this case taken
    # from peaks centroid (channel number x dwell time)
    fit_par, var = FitFuncCentroid(x_vals, y_vals, Centroid_guess, width, time_const)  # fit_par are fit parameters
    # whereas errors are filled in variance matrix (var)
    fit_error = np.sqrt(np.diag(var))
    for j in range(len(fit_par)):
        Peak_parameters[npeak, 2 * j] = fit_par[j]  # Filling peak parameters in columns
        Peak_parameters[npeak, 2 * j + 1] = fit_error[j]  # Filling error data in columns

    if Flag_Fig1 == 1:
        labelFit = 'Peak %d' % i  # Dynamic label maker
        plt.plot(x_vals, Gaus(x_vals, Peak_parameters[npeak, 0], Peak_parameters[npeak, 2], Peak_parameters[npeak, 4]),
                 color=next(colors), label=labelFit)  # Index 0,2,4 are Amplitude, Mean and Width

if Flag_Fig1 == 1:
    plt.legend(loc="upper right", fontsize=14)
    plt.xlim(15, 75)
    plt.xticks(fontsize=14)
    plt.xlabel(r'Time ($\mu$ s)', fontsize=18)
    plt.ylim(0, round(max(y_raw), -2))
    plt.yticks(fontsize=14)
    plt.ylabel('Counts', fontsize=18)
    plt.savefig('Fit.png')  # FIle saving


# Starting on width data
if Flag_Fig2 == 1:
    # Plot width vs Mean
    plt.figure(2, figsize=plt.figaspect(0.5), dpi=dpiCount)
    plt.plot(Peak_parameters[:, 2], Peak_parameters[:, 4], '--or')
#print(Peak_parameters[:,5])
# For removing phantom peaks
Peak_parameters_filtered_tpl=[]
#print(len(Peak_parameters))
#print(range(len(Peak_parameters)))
for i in range(len(Peak_parameters)):
    print(i)
    #print(Peak_parameters[i,5])
    if Peak_parameters[i,5] < 0.08:
        Peak_parameters_filtered_tpl.append(Peak_parameters[i,:])

        #print(Peak_parameters_filtered_tpl)
        Peak_parameters_filtered = np.array(Peak_parameters_filtered_tpl)


#print(Peak_parameters)
print(Peak_parameters_filtered)


# Starting on charge state
# Ion recorded in this run Cs
time_ratio = (Peak_parameters_filtered[:, 2] / Peak_parameters_filtered[0, 2]) ** 2  # time ratio calculated form experimental data
time_ratio_round = round_array(time_ratio, 2)  # Rounded to 2 decimal places but not used

# Created a loop to find minimum of deviation of charge state ratios and time ratios. The minimum of difference will
# give the best closest solution of charge states.
charge = np.arange(1, z_Cs)
charge_state_min = 1
charge_state_norm_index = 0
Res_arr = []
# Time ratio from experimental data
for charge_state_min in range(1, z_Cs):
    charge_state_max = charge_state_min + len(Peak_parameters_filtered) - 1
    charge_state_rev_arr = reverse_count(charge_state_min, charge_state_max)    # Charge array in reverse order created
    charge_state_arr = norm_charge(charge_state_rev_arr, charge_state_norm_index)   # Normalized charge state array
    msum = abs(sum(charge_state_arr - time_ratio))
    Res_arr.append(msum)

charge_state_min = Res_arr.index(min(Res_arr)) + 1
charge_state_max = charge_state_min + len(Peak_parameters_filtered) - 1
charge_state_rev_arr = reverse_count(charge_state_min, charge_state_max)  # Charge array in reverse order created
charge_state_arr = norm_charge(charge_state_rev_arr, charge_state_norm_index)  # Normalized charge state array

if Flag_Fig3 == 1:
    plt.figure(3, figsize=plt.figaspect(0.5), dpi=dpiCount)
    plt.plot(time_ratio, '--or', label='Time ratio - exp')
    plt.plot(charge_state_arr[charge_state_norm_index:], '--og', label='Charge ratio')
    # labelCharge = 'Charge states'
    # plt.plot(charge_arr, color=next(colors), label=labelCharge)
    plt.legend(loc="lower right", fontsize=14)

# Creating pandas dataframe (Peak_datasets) from numpy array (Peak_parameters)
Peak_data = pd.DataFrame({'Amplitude': Peak_parameters_filtered[:, 0], 'Amplitude_Error': Peak_parameters_filtered[:, 1],
                          'Mean': Peak_parameters_filtered[:, 2], 'Mean_Error': Peak_parameters_filtered[:, 3],
                          'Width': Peak_parameters_filtered[:, 4], 'Width_Error': Peak_parameters_filtered[:, 5],
                          'Time ratio': time_ratio[:]})

# Showing data on screen terminal and writing file to external csv.
print(Peak_data)
Peak_data.to_csv('Fit_vals.csv', index=False)

SSM = sum((charge_state_arr - time_ratio)**2)

print("Least sum is %f " % SSM)


# Plotting final spectra with charge identified peaks.
plt.figure(4, figsize=plt.figaspect(0.5), dpi=dpiCount)
plt.plot(x_vals, y_vals, '-k', Linewidth=1, label="Data")

charge_peak = charge_state_max
for npeak in range(len(Peak_parameters_filtered)):  # First level for rows
    if Flag_Fig4 == 1:
        labelFit = '%d+' % charge_peak  # Dynamic label maker
        plt.plot(x_vals, Gaus(x_vals, Peak_parameters_filtered[npeak, 0], Peak_parameters_filtered[npeak, 2], Peak_parameters_filtered[npeak, 4]),
                 color=next(colors), label=labelFit)  # Index 0,2,4 are Amplitude, Mean and Width
        charge_peak -= 1

plt.legend(loc="upper right", fontsize=14)
plt.xlim(15, 75)
plt.xticks(fontsize=14)
plt.xlabel(r'Time ($\mu$ s)', fontsize=18)
plt.ylim(0, round(max(y_raw), -2))
plt.yticks(fontsize=14)
plt.ylabel('Counts', fontsize=18)
plt.savefig('Fit.png')  # FIle saving
plt.show()  # This should be at the end so that all the plt objects show
