from tools import lorentzian
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from pathlib import Path
from reader import reader
from tabulate import tabulate
from tools import calibration_Ch0, calibration_Ch1, calibration_Ch2, error_Ch0, error_Ch1, error_Ch2, inverse_calibration_Ch0, inverse_calibration_Ch1, inverse_calibration_Ch2
from tools import inverse_error_Ch0
from tools import gaussian
import os
import pandas as pd

# Show
SHOW = False

# Save Pictures 
SAVE = False

# Put Titles
TITLES = False

# Define paths
# data
DATA_PATH = Path('data')
# data folders
TT_20_PATH = DATA_PATH / 'TT_20' / 'UNFILTERED'
TT_21_PATH = DATA_PATH / 'TT_21' / 'UNFILTERED'
TT_22_PATH = DATA_PATH / 'TT_22' / 'UNFILTERED'
TT_23_PATH = DATA_PATH / 'TT_23' / 'UNFILTERED'
TT_24_PATH = DATA_PATH / 'TT_24' / 'UNFILTERED'
TT_25_PATH = DATA_PATH / 'TT_25' / 'UNFILTERED'
TT_26_PATH = DATA_PATH / 'TT_26' / 'UNFILTERED'
TT_27_PATH = DATA_PATH / 'TT_27' / 'UNFILTERED'
# image
IMAGE_PATH = Path('images')
IMAGE_PATH.mkdir(exist_ok=True, parents=True)

# Reading the data files

# -----------------------------------------------------------------------------
# Sample with Calibration
# -----------------------------------------------------------------------------
# TT_21
Cal_Chn0 = reader(TT_21_PATH / 'CH0@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 0)
Cal_Chn1 = reader(TT_21_PATH / 'CH1@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 1)
Cal_Chn2 = reader(TT_21_PATH / 'CH2@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 2)


# -----------------------------------------------------------------------------
# Sample with LiF
# -----------------------------------------------------------------------------
# TT_23
LIF_Chn0 = reader(TT_23_PATH / 'CH0@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 0)
LIF_Chn1 = reader(TT_23_PATH / 'CH1@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 1)
LIF_Chn2 = reader(TT_23_PATH / 'CH2@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 2)

# -----------------------------------------------------------------------------
# Sample with LiAlO2
# -----------------------------------------------------------------------------
# TT_24
LiAlO2_Chn0 = reader(TT_24_PATH / 'CH0@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 0)
LiAlO2_Chn1 = reader(TT_24_PATH / 'CH1@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 1)
LiAlO2_Chn2 = reader(TT_24_PATH / 'CH2@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 2)

# -----------------------------------------------------------------------------
# Sample with implantation of Li in Al
# -----------------------------------------------------------------------------
# TT_26
Implanted_Chn0 = reader(TT_26_PATH / 'CH0@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 0)
Implanted_Chn1 = reader(TT_26_PATH / 'CH1@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 1)
Implanted_Chn2 = reader(TT_26_PATH / 'CH2@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 2)

# -----------------------------------------------------------------------------
# From now on we will use only the data from channel 0
# -----------------------------------------------------------------------------
def Detector_Calibration(detector : int = 0, bins : int = 1024) -> np.ndarray:
    x = np.linspace(0, bins, bins)
    if detector == 0:
        return calibration_Ch0(x)
    elif detector == 1:
        return calibration_Ch1(x)
    elif detector == 2:
        return calibration_Ch2(x)
    else:
        raise ValueError('Invalid detector number')

# -----------------------------------------------------------------------------
# Firstly we are overlap the data with the calibration
# We know that the Lithium peaks should have more energy than the peaks of the
# calibration, so if we see peaks above the calibration peaks, we know that
# it might be Lithium
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 10))
plt.plot(LIF_Chn0, label='LiF', alpha=0.5)
plt.plot(LiAlO2_Chn0, label='LiAlO2', alpha=0.5)
plt.plot(Implanted_Chn0, label='Implanted', alpha=0.5)
plt.plot(Cal_Chn0, label='Calibration')
plt.legend()
if TITLES: plt.title('Overlap')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.xlim(0, 1024)
if SAVE: plt.savefig(IMAGE_PATH / 'OverlapCalib.png', dpi=600)
plt.grid()
if SHOW: plt.show()


# -----------------------------------------------------------------------------
# Now we do the calibration
# -----------------------------------------------------------------------------
# os.system('python3 calib.py')

# -----------------------------------------------------------------------------
# Analyzing the alpha peaks
# -----------------------------------------------------------------------------
alphas = pd.read_csv(DATA_PATH / 'alphas.csv') # Load the data
x = Detector_Calibration(detector=0, bins=1024) # Define the x range
plt.figure(figsize=(10, 10))
plt.vlines(alphas['Energy'], 0, alphas['Percentage'], color='r')
# Put text on the peaks
for i in range(len(alphas)):
    text = str(i)
    plt.text(alphas['Energy'][i], alphas['Percentage'][i], r'$\alpha_{' + text + '}$', fontsize=15)
if TITLES: plt.title('Alpha peaks')
plt.xlabel('Energy [keV]')
plt.ylabel('Percentage (%)')
plt.xlim(0, 13000)
if SAVE: plt.savefig(IMAGE_PATH / 'AlphaPeaks.png', dpi=600)
plt.grid()
if SHOW: plt.show()

# -----------------------------------------------------------------------------
# Draw the Lorentzian fit for the peaks
# -----------------------------------------------------------------------------
# We will use the Lorentzian function defined in tools.py
plt.figure(figsize=(10, 10))
x = np.linspace(0, 2048, 2048)
x = calibration_Ch0(x)
for i in range(len(alphas)):
    plt.plot(x, alphas['Percentage'][i] * lorentzian(x,  alphas['Energy'][i], alphas['Band'][i]), label='Lorentzian', marker='', linestyle='--')
if TITLES: plt.title('Lorentzian function')
# Put text on the peaks
for i in range(len(alphas)):
    text = str(i)
    plt.text(alphas['Energy'][i], alphas['Percentage'][i] * lorentzian(alphas['Energy'][i],  alphas['Energy'][i], alphas['Band'][i]), r'$\alpha_{' + text + '}$', fontsize=15)
    plt.vlines(alphas['Energy'][i], 0, alphas['Percentage'][i] * lorentzian(alphas['Energy'][i],  alphas['Energy'][i], alphas['Band'][i]), color='r', label='Alpha peaks')
if TITLES: plt.title('Alpha peaks')
# plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('Percentage (%)')
plt.xlim(0, 13000)
plt.legend()
if SAVE: plt.savefig(IMAGE_PATH / 'PeaksLorentzian.png', dpi=600)
plt.grid()
if SHOW: plt.show()

# -----------------------------------------------------------------------------
# Overlap the alpha lines with the data to confirm Lithium
# -----------------------------------------------------------------------------
# LiF
plt.figure(figsize=(10, 10))
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0(x)
plt.plot(x, LIF_Chn0, label='LiF', color='b', alpha=0.5)
for i in range(len(alphas)):
    plt.vlines(alphas['Energy'][i], 0, max(LIF_Chn0), color='r')
    plt.text(alphas['Energy'][i], np.mean(LIF_Chn0) * 0.9, r'$\alpha_{' + str(i) + '}$', fontsize=15)

# Guassian for Channel 0
x = np.array(range(830, 950))
y = LIF_Chn0[830:950]
# curve fit
popt0, pcov0 = curve_fit(gaussian, x, y, p0=[100, 850, 10])
errors0 = np.sqrt(np.diag(pcov0))

# Plot a line on the x0 
plt.vlines(calibration_Ch0(popt0[1]), 0, max(LIF_Chn0), color='g', label=r'Experimental $a_0$', linestyle='--')

# Print on terminal the results
print(tabulate([['Sample with LiF']], tablefmt='fancy_grid'))

info = [
    ['Detector', 'a (Counts)', 'x0 [Channels]', 'sigma [Channels]', 'x0 [keV]', 'sigma [keV]', 'Error from Teoretical x0 [keV] (%)'],
    ['0', f'{popt0[0]:.2f} ± {errors0[0]:.2f}', f'{popt0[1]:.2f} ± {errors0[1]:.2f}', f'{popt0[2]:.2f} ± {errors0[2]:.2f}', f'{calibration_Ch0(popt0[1]):.2f} ± {error_Ch0(popt0[1], errors0[1]):.2f}', f'{calibration_Ch0(popt0[2]):.2f} ± {error_Ch0(popt0[2], errors0[2]):.2f}', f'{(calibration_Ch0(popt0[1]) - alphas["Energy"][0]) / alphas["Energy"][0] * 100:.2f}'],
]

print(tabulate(info, headers='firstrow', tablefmt='fancy_grid'))

# Print table for LaTeX
print('Code for LaTeX\n\n')
print(tabulate(info, headers='firstrow', tablefmt='latex'), '\n\n')

plt.legend()
if TITLES: plt.title('Calibration')
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
plt.xlim(0, calibration_Ch0(1024))
plt.ylim(0.1, max(LIF_Chn0))
if SAVE: plt.savefig(IMAGE_PATH / 'OverlapLiF.png', dpi=600)
plt.grid()
if SHOW: plt.show()

# LiAlO2
plt.figure(figsize=(10, 10))
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0(x)
plt.plot(x, LiAlO2_Chn0, label='LiAlO2', color='orange', alpha=0.5)
for i in range(len(alphas)):
    plt.vlines(alphas['Energy'][i], 0, max(LiAlO2_Chn0), color='r')
    plt.text(alphas['Energy'][i], np.mean(LiAlO2_Chn0) * 0.9, r'$\alpha_{' + str(i) + '}$', fontsize=15)

# Guassian for Channel 0
# Channel 0
x = np.array(range(830, 950))
y = LiAlO2_Chn0[830:950]
# curve fit
popt0, pcov0 = curve_fit(gaussian, x, y, p0=[40, 850, 10])
errors0 = np.sqrt(np.diag(pcov0))

# Plot a line on the x0
plt.vlines(calibration_Ch0(popt0[1]), 0, max(LiAlO2_Chn0), color='g', label=r'Experimental $a_0$', linestyle='--')

# Print on terminal the results
print(tabulate([['Sample with LiAlO2']], tablefmt='fancy_grid'))

info = [
    ['Detector', 'a (Counts)', 'x0 [Channels]', 'sigma [Channels]', 'x0 [keV]', 'sigma [keV]', 'Error from Teoretical x0 [keV] (%)'],
    ['0', f'{popt0[0]:.2f} ± {errors0[0]:.2f}', f'{popt0[1]:.2f} ± {errors0[1]:.2f}', f'{popt0[2]:.2f} ± {errors0[2]:.2f}', f'{calibration_Ch0(popt0[1]):.2f} ± {error_Ch0(popt0[1], errors0[1]):.2f}', f'{calibration_Ch0(popt0[2]):.2f} ± {error_Ch0(popt0[2], errors0[2]):.2f}', f'{(calibration_Ch0(popt0[1]) - alphas["Energy"][0]) / alphas["Energy"][0] * 100:.2f}'],
]

print(tabulate(info, headers='firstrow', tablefmt='fancy_grid'))

# Print table for LaTeX
print('Code for LaTeX\n\n')
print(tabulate(info, headers='firstrow', tablefmt='latex'), '\n\n')

if TITLES: plt.title('Calibration')
plt.legend()
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
plt.xlim(0, calibration_Ch0(1024))
plt.ylim(0.1, max(LiAlO2_Chn0))
if SAVE: plt.savefig(IMAGE_PATH / 'OverlapLiAlO2.png', dpi=600)
plt.grid()
if SHOW: plt.show()

# Implanted
plt.figure(figsize=(10, 10))
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0(x)
plt.plot(x, Implanted_Chn0, label='Implanted', color='g', alpha=0.5)
for i in range(len(alphas)):
    plt.vlines(alphas['Energy'][i], 0, max(Implanted_Chn0), color='r')
    plt.text(alphas['Energy'][i], np.mean(Implanted_Chn0) * 0.9, r'$\alpha_{' + str(i) + '}$', fontsize=15)

# Guassian for Channel 0
x = np.array(range(800, 900))
y = Implanted_Chn0[800:900]
# curve fit
popt0, pcov0 = curve_fit(gaussian, x, y, p0=[12, 850, 10])
errors0 = np.sqrt(np.diag(pcov0))

# Plot a line on the x0
plt.vlines(calibration_Ch0(popt0[1]), 0, max(Implanted_Chn0), color='g', label=r'Experimental $a_0$', linestyle='--')

# Print on terminal the results
print(tabulate([['Sample with Implanted']], tablefmt='fancy_grid'), '\n\n')

info = [
    ['Detector', 'a (Counts)', 'x0 [Channels]', 'sigma [Channels]', 'x0 [keV]', 'sigma [keV]', 'Error from Teoretical x0 [keV] (%)'],
    ['0', f'{popt0[0]:.2f} ± {errors0[0]:.2f}', f'{popt0[1]:.2f} ± {errors0[1]:.2f}', f'{popt0[2]:.2f} ± {errors0[2]:.2f}', f'{calibration_Ch0(popt0[1]):.2f} ± {error_Ch0(popt0[1], errors0[1]):.2f}', f'{calibration_Ch0(popt0[2]):.2f} ± {error_Ch0(popt0[2], errors0[2]):.2f}', f'{(calibration_Ch0(popt0[1]) - alphas["Energy"][0]) / alphas["Energy"][0] * 100:.2f}'],
]

print(tabulate(info, headers='firstrow', tablefmt='fancy_grid'))

# Print table for LaTeX
print('Code for LaTeX\n\n')
print(tabulate(info, headers='firstrow', tablefmt='latex'), '\n\n')

plt.legend()
if TITLES: plt.title('Calibration')
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
plt.xlim(0, calibration_Ch0(1024))
plt.ylim(0.1, max(Implanted_Chn0))
if SAVE: plt.savefig(IMAGE_PATH / 'OverlapImplanted.png', dpi=600)
plt.grid()
if SHOW: plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Confirmed the Presence of Lithium in the samples
# We can obtain the energy of the Experimental Lithium peaks
# and compare with the theoretical values
# ----------------------------------------------------------------------------------------------------------------------
# os.system('python3 analise.py')


# ----------------------------------------------------------------------------------------------------------------------
# Now we will analyze the rest of the peaks
# ----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LiF
# -----------------------------------------------------------------------------
# Define the x range
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0(x)

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 10))

# Plot data
ax.plot(x, LIF_Chn0, label='LiF', color='b', alpha=0.5)

# ---------------------------------------
# Electronic Noise
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (11, 729)  # (x, y) coordinates of the lower left corner
width = 150                    # Width of the rectangle
height = 730000                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='orange', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_one = patches.Patch(color='orange', label='Electronic Noise')

# ---------------------------------------
# ???
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (250, 830)  # (x, y) coordinates of the lower left corner
width = 1150                    # Width of the rectangle
height = 17000                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='blue', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_two = patches.Patch(color='blue', label='Backscattering')

# ---------------------------------------
# Pileup
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (1252, 1)  # (x, y) coordinates of the lower left corner
width = 1300                    # Width of the rectangle
height = 234                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='red', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_three = patches.Patch(color='red', label='Pile Up')

# ---------------------------------------
# ???
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (3000, 1)  # (x, y) coordinates of the lower left corner
width = 5500                    # Width of the rectangle
height = 500                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='green', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_four = patches.Patch(color='green', label='Your Rectangle Label')

# Log Scale in y axis
ax.set_yscale('log')


# ---------------------------------------
# Backscattering Barrier
# ---------------------------------------
# Choose Points to draw the line
x1 = 988
x2 = 1305

# Plot the line
plt.vlines(np.mean([x1, x2]), 0, max(LIF_Chn0), color='r', label='Backscattering Barrier')
plt.text(np.mean([x1, x2]) *1.2, max(LIF_Chn0) * 1e-1, f'{np.mean([x1, x2])} keV', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Plot the x1 and x2 points
plt.plot(x1, LIF_Chn0[int(inverse_calibration_Ch0(x1))], 'o', color='r')
plt.plot(x2, LIF_Chn0[int(inverse_calibration_Ch0(x2))], 'o', color='r')


# Add legend for both line and rectangle
ax.legend(handles=[rect_patch_one, rect_patch_two, rect_patch_three, rect_patch_four, plt.Line2D([], [], color='b', label='LiAlO2'), plt.Line2D([], [], color='r', label='Backscattering Barrier')])
# Show plot
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
if SAVE: plt.savefig(IMAGE_PATH / 'FullAnalysisLiF.png', dpi=600)
plt.grid()
if SHOW: plt.show()


# -----------------------------------------------------------------------------
# LiAlO2
# -----------------------------------------------------------------------------
# Define the x range
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0(x)

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 10))

# Plot data
ax.plot(x, LiAlO2_Chn0, label='LiAlO2', color='orange', alpha=0.5)

# ---------------------------------------
# Electronic Noise
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (11, 729)  # (x, y) coordinates of the lower left corner
width = 100                    # Width of the rectangle
height = 730000                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='orange', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_one = patches.Patch(color='orange', label='Electronic Noise')

# ---------------------------------------
# ???
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (250, 830)  # (x, y) coordinates of the lower left corner
width = 1150                    # Width of the rectangle
height = 19000                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='blue', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_two = patches.Patch(color='blue', label='Backscattering')

# ---------------------------------------
# Pileup
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (1252, 1)  # (x, y) coordinates of the lower left corner
width = 1000                    # Width of the rectangle
height = 234                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='red', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_three = patches.Patch(color='red', label='Pile Up')

# ---------------------------------------
# ???
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (3000, 1)  # (x, y) coordinates of the lower left corner
width = 5500                    # Width of the rectangle
height = 100                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='green', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_four = patches.Patch(color='green', label='Your Rectangle Label')

# Log Scale in y axis
ax.set_yscale('log')

# ---------------------------------------
# Backscattering Barrier
# ---------------------------------------
# Choose Points to draw the line
x1 = 1033
x2 = 1267

# Plot the line
plt.vlines(np.mean([x1, x2]), 0, max(LiAlO2_Chn0), color='r', label='Backscattering Barrier')
plt.text(np.mean([x1, x2]) *1.2, max(LiAlO2_Chn0) * 1e-1, f'{np.mean([x1, x2])} keV', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Plot the x1 and x2 points
plt.plot(x1, LiAlO2_Chn0[int(inverse_calibration_Ch0(x1))], 'o', color='r')
plt.plot(x2, LiAlO2_Chn0[int(inverse_calibration_Ch0(x2))], 'o', color='r')

# Add legend for both line and rectangle
ax.legend(handles=[rect_patch_one, rect_patch_two, rect_patch_three, rect_patch_four, plt.Line2D([], [], color='b', label='LiAlO2'), plt.Line2D([], [], color='r', label='Backscattering Barrier')])

# Show plot
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
if SAVE: plt.savefig(IMAGE_PATH / 'FullAnalysisLiAlO2.png', dpi=600)
plt.grid()
if SHOW: plt.show()


# -----------------------------------------------------------------------------
# Implanted
# -----------------------------------------------------------------------------
# Define the x range
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0(x)

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 10))

# Plot data
ax.plot(x, Implanted_Chn0, label='Implanted', color='green', alpha=0.5)

# ---------------------------------------
# Electronic Noise
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (11, 5000)  # (x, y) coordinates of the lower left corner
width = 100                    # Width of the rectangle
height = 730000                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='orange', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_one = patches.Patch(color='orange', label='Electronic Noise')

# ---------------------------------------
# ???
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (250, 830)  # (x, y) coordinates of the lower left corner
width = 1150                    # Width of the rectangle
height = 187000                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='blue', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_two = patches.Patch(color='blue', label='Backscattering')

# ---------------------------------------
# Pileup
# ---------------------------------------
# Parameters for the rectangle
lower_left_corner = (1252, 1)  # (x, y) coordinates of the lower left corner
width = 1200                    # Width of the rectangle
height = 734                   # Height of the rectangle

# Draw a Rectangle patch (Circle will shrink due to the axis scale)
rectangle = patches.Rectangle(lower_left_corner, width, height, edgecolor='red', facecolor='none')
ax.add_patch(rectangle)

# Create a legend handle for the rectangle
rect_patch_three = patches.Patch(color='red', label='Pile Up')

# Log Scale in y axis
ax.set_yscale('log')

# ---------------------------------------
# Backscattering Barrier
# ---------------------------------------
# Choose Points to draw the line
x1 = 1113
x2 = 1308

# Plot the line
plt.vlines(np.mean([x1, x2]), 0, max(Implanted_Chn0), color='r', label='Backscattering Barrier')
plt.text(np.mean([x1, x2]) *1.2, max(Implanted_Chn0) * 1e-1, f'{np.mean([x1, x2])} keV', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Plot the x1 and x2 points
plt.plot(x1, Implanted_Chn0[int(inverse_calibration_Ch0(x1))], 'o', color='r')
plt.plot(x2, Implanted_Chn0[int(inverse_calibration_Ch0(x2))], 'o', color='r')

# Add legend for both line and rectangle
ax.legend(handles=[rect_patch_one, rect_patch_two, rect_patch_three, rect_patch_four, plt.Line2D([], [], color='b', label='LiAlO2'), plt.Line2D([], [], color='r', label='Backscattering Barrier')])

# Show plot
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
if SAVE: plt.savefig(IMAGE_PATH / 'FullAnalysisImplanted.png', dpi=600)
plt.grid()
if SHOW: plt.show()





# ----------------------------------------------------------------------------------------------------------------------
# Since the Implantation sample is the one with the smoothest surface, we can use it's backscattering barrier
# to improve our calibration. Henceforth, reducing the error of the calibration for smaller energies.
# The other samples were not used because they have a rougher surface, which makes it harder to identify the
# exact backscattering barrier and compare with the theoretical value.
# ----------------------------------------------------------------------------------------------------------------------
print(tabulate([['Lithium Peaks with Calibration Using an Extra Point from the Aluminium Barrier']], tablefmt='fancy_grid'), '\n\n')
df = pd.read_csv(DATA_PATH / 'calibration_points_Detector0.csv')
EnergyBackscatteringBarrier = pd.read_csv(DATA_PATH / 'backscattering.csv')

Xpoints = df['Points'].to_numpy()
Ypoints = df['Energy'].to_numpy()
# Append the backscattering barrier
Xpoints = np.append(Xpoints, inverse_calibration_Ch0(np.mean([x1, x2])))
Ypoints = np.append(Ypoints, EnergyBackscatteringBarrier['Energy'][0])

# Function to fit the calibration
def linear(x, m, c):
    return m * x + c

# Calibração (channel 0)
params0, params_cov0 = curve_fit(linear, Xpoints, Ypoints)
errors0 = np.sqrt(np.diag(params_cov0))
m0 = params0[0]
c0 = params0[1]

plt.figure(figsize=(10, 10))
plt.plot(Xpoints, Ypoints, 'o', label='data')
plt.plot(Xpoints, linear(Xpoints, *params0), label='fit')
if TITLES: plt.title('Calibration Channel 0')
plt.legend()
plt.xlabel('Channel')
plt.ylabel('Energy [keV]')
plt.text(260,
         5600,
         f'E = ({m0:.2f} ± {errors0[0]:.2f}) * Chn + ({c0:.2f} ± {errors0[1]:.2f})',
         fontsize=10,
         bbox=dict(facecolor='white',alpha=0.5))
plt.grid()
if SAVE: plt.savefig(IMAGE_PATH / 'Chn0_calib_with_Backscattering.png', dpi=600)
plt.grid()
if SHOW: plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Reanalyze the data with the new calibration
# ----------------------------------------------------------------------------------------------------------------------
calibration_Ch0_Backscattering = lambda x : m0 * x + c0
# LiF
plt.figure(figsize=(10, 10))
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0_Backscattering(x)
plt.plot(x, LIF_Chn0, label='LiF', color='b', alpha=0.5)
for i in range(len(alphas)):
    plt.vlines(alphas['Energy'][i], 0, max(LIF_Chn0), color='r')
    plt.text(alphas['Energy'][i], np.mean(LIF_Chn0) * 0.9, r'$\alpha_{' + str(i) + '}$', fontsize=15)

# Guassian for Channel 0
x = np.array(range(830, 950))
y = LIF_Chn0[830:950]
# curve fit
popt0, pcov0 = curve_fit(gaussian, x, y, p0=[100, 850, 10])
errors0 = np.sqrt(np.diag(pcov0))

# Plot a line on the x0 
plt.vlines(calibration_Ch0_Backscattering(popt0[1]), 0, max(LIF_Chn0), color='g', label=r'Experimental $a_0$', linestyle='--')

# Print on terminal the results
print(tabulate([['Sample with LiF']], tablefmt='fancy_grid'))

info = [
    ['Detector', 'a (Counts)', 'x0 [Channels]', 'sigma [Channels]', 'x0 [keV]', 'sigma [keV]', 'Error from Teoretical x0 [keV] (%)'],
    ['0', f'{popt0[0]:.2f} ± {errors0[0]:.2f}', f'{popt0[1]:.2f} ± {errors0[1]:.2f}', f'{popt0[2]:.2f} ± {errors0[2]:.2f}', f'{calibration_Ch0_Backscattering(popt0[1]):.2f} ± {error_Ch0(popt0[1], errors0[1]):.2f}', f'{calibration_Ch0_Backscattering(popt0[2]):.2f} ± {error_Ch0(popt0[2], errors0[2]):.2f}', f'{(calibration_Ch0_Backscattering(popt0[1]) - alphas["Energy"][0]) / alphas["Energy"][0] * 100:.2f}'],
]

print(tabulate(info, headers='firstrow', tablefmt='fancy_grid'))

# Print table for LaTeX
print('Code for LaTeX\n\n')
print(tabulate(info, headers='firstrow', tablefmt='latex'), '\n\n')

plt.legend()
if TITLES: plt.title('Calibration')
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
plt.xlim(0, calibration_Ch0_Backscattering(1024))
plt.ylim(0.1, max(LIF_Chn0))
if SAVE: plt.savefig(IMAGE_PATH / 'OverlapLiF_NC.png', dpi=600)
plt.grid()
if SHOW: plt.show()

# LiAlO2
plt.figure(figsize=(10, 10))
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0_Backscattering(x)
plt.plot(x, LiAlO2_Chn0, label='LiAlO2', color='orange', alpha=0.5)
for i in range(len(alphas)):
    plt.vlines(alphas['Energy'][i], 0, max(LiAlO2_Chn0), color='r')
    plt.text(alphas['Energy'][i], np.mean(LiAlO2_Chn0) * 0.9, r'$\alpha_{' + str(i) + '}$', fontsize=15)

# Guassian for Channel 0
# Channel 0
x = np.array(range(830, 950))
y = LiAlO2_Chn0[830:950]
# curve fit
popt0, pcov0 = curve_fit(gaussian, x, y, p0=[40, 850, 10])
errors0 = np.sqrt(np.diag(pcov0))

# Plot a line on the x0
plt.vlines(calibration_Ch0_Backscattering(popt0[1]), 0, max(LiAlO2_Chn0), color='g', label=r'Experimental $a_0$', linestyle='--')

# Print on terminal the results
print(tabulate([['Sample with LiAlO2']], tablefmt='fancy_grid'))

info = [
    ['Detector', 'a (Counts)', 'x0 [Channels]', 'sigma [Channels]', 'x0 [keV]', 'sigma [keV]', 'Error from Teoretical x0 [keV] (%)'],
    ['0', f'{popt0[0]:.2f} ± {errors0[0]:.2f}', f'{popt0[1]:.2f} ± {errors0[1]:.2f}', f'{popt0[2]:.2f} ± {errors0[2]:.2f}', f'{calibration_Ch0_Backscattering(popt0[1]):.2f} ± {error_Ch0(popt0[1], errors0[1]):.2f}', f'{calibration_Ch0_Backscattering(popt0[2]):.2f} ± {error_Ch0(popt0[2], errors0[2]):.2f}', f'{(calibration_Ch0_Backscattering(popt0[1]) - alphas["Energy"][0]) / alphas["Energy"][0] * 100:.2f}'],
]

print(tabulate(info, headers='firstrow', tablefmt='fancy_grid'))

# Print table for LaTeX
print('Code for LaTeX\n\n')
print(tabulate(info, headers='firstrow', tablefmt='latex'), '\n\n')

if TITLES: plt.title('Calibration')
plt.legend()
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
plt.xlim(0, calibration_Ch0_Backscattering(1024))
plt.ylim(0.1, max(LiAlO2_Chn0))
if SAVE: plt.savefig(IMAGE_PATH / 'OverlapLiAlO2_NC.png', dpi=600)
plt.grid()
if SHOW: plt.show()

# Implanted
plt.figure(figsize=(10, 10))
x = np.linspace(0, 1024, 1024)
x = calibration_Ch0_Backscattering(x)
plt.plot(x, Implanted_Chn0, label='Implanted', color='g', alpha=0.5)
for i in range(len(alphas)):
    plt.vlines(alphas['Energy'][i], 0, max(Implanted_Chn0), color='r')
    plt.text(alphas['Energy'][i], np.mean(Implanted_Chn0) * 0.9, r'$\alpha_{' + str(i) + '}$', fontsize=15)

# Guassian for Channel 0
x = np.array(range(800, 900))
y = Implanted_Chn0[800:900]
# curve fit
popt0, pcov0 = curve_fit(gaussian, x, y, p0=[12, 850, 10])
errors0 = np.sqrt(np.diag(pcov0))

# Plot a line on the x0
plt.vlines(calibration_Ch0_Backscattering(popt0[1]), 0, max(Implanted_Chn0), color='g', label=r'Experimental $a_0$', linestyle='--')

# Print on terminal the results
print(tabulate([['Sample with Implanted']], tablefmt='fancy_grid'))

info = [
    ['Detector', 'a (Counts)', 'x0 [Channels]', 'sigma [Channels]', 'x0 [keV]', 'sigma [keV]', 'Error from Teoretical x0 [keV] (%)'],
    ['0', f'{popt0[0]:.2f} ± {errors0[0]:.2f}', f'{popt0[1]:.2f} ± {errors0[1]:.2f}', f'{popt0[2]:.2f} ± {errors0[2]:.2f}', f'{calibration_Ch0_Backscattering(popt0[1]):.2f} ± {error_Ch0(popt0[1], errors0[1]):.2f}', f'{calibration_Ch0_Backscattering(popt0[2]):.2f} ± {error_Ch0(popt0[2], errors0[2]):.2f}', f'{(calibration_Ch0_Backscattering(popt0[1]) - alphas["Energy"][0]) / alphas["Energy"][0] * 100:.2f}'],
]

print(tabulate(info, headers='firstrow', tablefmt='fancy_grid'))

# Print table for LaTeX
print('Code for LaTeX\n\n')
print(tabulate(info, headers='firstrow', tablefmt='latex'), '\n\n')

plt.legend()
if TITLES: plt.title('Calibration')
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts')
plt.xlim(0, calibration_Ch0_Backscattering(1024))
plt.ylim(0.1, max(Implanted_Chn0))
if SAVE: plt.savefig(IMAGE_PATH / 'OverlapImplanted_NC.png', dpi=600)
plt.grid()
if SHOW: plt.show()