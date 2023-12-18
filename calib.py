import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from reader import reader
import pandas as pd

# Define paths
# For the calibration we will use the data from the TT_21 file
# data
DATA_PATH = Path('data')
# data folders
TT_21_PATH = DATA_PATH / 'TT_21' / 'UNFILTERED'
# image
IMAGE_PATH = Path('images')
IMAGE_PATH.mkdir(exist_ok=True, parents=True)

# TT_21
TT21_Chn0 = reader(TT_21_PATH/'CH0@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 0)
TT21_Chn1 = reader(TT_21_PATH/'CH1@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 1)
TT21_Chn2 = reader(TT_21_PATH/'CH2@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 2)

# TT_21
plt.figure()
plt.plot(TT21_Chn0, label='Chnannel 0')
plt.plot(TT21_Chn1, label='Channel 1')
plt.plot(TT21_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_21')
plt.yscale('log')
plt.xlabel('Channel')
plt.xlim(0, 1024)
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_21.png')
plt.show()

# Calibration
# Define CSV file to keep the calibration data
CALIB_PATH = Path('data')
CALIB_PATH.mkdir(exist_ok=True, parents=True)
CALIB_FILE = CALIB_PATH / 'calibration.csv'
df = pd.DataFrame(columns=['m', 'c', 'error_m', 'error_c'])


# Gaussian fit for the peaks (soma 3 de gaussianas)
def gaussian(x, a1, x01, sigma1, a2, x02, sigma2, a3, x03, sigma3, c):
    return a1*np.exp(-(x-x01)**2/(2*sigma1**2)) + a2*np.exp(-(x-x02)**2/(2*sigma2**2)) + a3*np.exp(-(x-x03)**2/(2*sigma3**2)) + c

# Channel 0
x = np.array(range(545, 650))
y = TT21_Chn0[545:650]
# curve fit
popt0, pcov0 = curve_fit(gaussian, x, y, p0=[90, 556, 5, 70, 593, 5, 70, 627, 5, 0])

plt.figure()
plt.plot(TT21_Chn0, label='Chnannel 0')
plt.plot(x, gaussian(x, *popt0), label='Chnannel 0 fit')
plt.title('Fit channel 0')
plt.legend()
plt.xlabel('Channel')
plt.yscale('log')
plt.ylabel('Counts')
plt.xlim(500, 700)
plt.ylim(1, 1e2)
plt.savefig(IMAGE_PATH/'TT_21_Chn0.png')
plt.show()

# Channel 1
x = np.array(range(545, 700))
y = TT21_Chn1[545:700]
# curve fit
popt1, pcov1 = curve_fit(gaussian, x, y, p0=[90, 565, 5, 70, 605, 5, 70, 640, 5, 0])

plt.figure()
plt.plot(TT21_Chn1, label='Chnannel 1')
plt.plot(x, gaussian(x, *popt1), label='Chnannel 1 fit')
plt.title('Fit channel 1')
plt.legend()
plt.xlabel('Channel')
plt.yscale('log')
plt.ylabel('Counts')
plt.xlim(500, 700)
plt.ylim(1, 1e2)
plt.savefig(IMAGE_PATH/'TT_21_Chn1.png')
plt.show()

# Channel 2
x = np.array(range(545, 700))
y = TT21_Chn2[545:700]
# curve fit
popt2, pcov2 = curve_fit(gaussian, x, y, p0=[90, 565, 5, 70, 605, 5, 70, 640, 5, 0])

plt.figure()
plt.plot(TT21_Chn2, label='Chnannel 2')
plt.plot(x, gaussian(x, *popt2), label='Chnannel 2 fit')
plt.title('Fit channel 2')
plt.legend()
plt.xlabel('Channel')
plt.yscale('log')
plt.ylabel('Counts')
plt.xlim(500, 700)
plt.ylim(1, 1e2)
plt.savefig(IMAGE_PATH/'TT_21_Chn2.png')
plt.show()


# valores de energia para os 3 picos
energy = np.array([5156.59, 5485.56, 5804.82])

# média das gaussianas
x0 = np.array([popt0[1], popt0[4], popt0[7]])
x1 = np.array([popt1[1], popt1[4], popt1[7]])
x2 = np.array([popt2[1], popt2[4], popt2[7]])

sigma0 = np.array([popt0[2], popt0[5], popt0[8]])
sigma1 = np.array([popt1[2], popt1[5], popt1[8]])
sigma2 = np.array([popt2[2], popt2[5], popt2[8]])

def linear(x, m, c):
    return m * x + c

a1 = np.array([popt0[0], popt0[3], popt0[6]])
a2 = np.array([popt1[0], popt1[3], popt1[6]])
a3 = np.array([popt2[0], popt2[3], popt2[6]])

# Calibração (channel 0)
params0, params_cov0 = curve_fit(linear, x0, energy)
errors0 = np.sqrt(np.diag(params_cov0))
m0 = params0[0]
c0 = params0[1]

plt.figure()
plt.plot(x0, energy, 'o', label='data')
plt.plot(x0, linear(x0, *params0), label='fit')
plt.title('Calibration channel 0')
plt.legend()
plt.xlabel('Channel')
plt.ylabel('Energy (keV)')
plt.text(560, 5600, f'E = ({m0:.2f} ± {errors0[0]:.2f}) * Chn + {c0:.2f} ± {errors0[1]:.2f}', fontsize=10)
plt.savefig(IMAGE_PATH/'TT_21_Chn0_calib.png')
plt.show()

print(f'Calibração_0: E = ({m0:.2f} ± {errors0[0]:.2f}) * Chn + {c0:.2f} ± {errors0[1]:.2f}')

# Calibração (channel 1)
params1, params_cov1 = curve_fit(linear, x1, energy)
errors1 = np.sqrt(np.diag(params_cov1))
m1 = params1[0]
c1 = params1[1]

plt.figure()
plt.plot(x1, energy, 'o', label='data')
plt.plot(x1, linear(x1, *params1), label='fit')
plt.title('Calibration channel 1')
plt.legend()
plt.xlabel('Channel')
plt.ylabel('Energy (keV)')
plt.text(570, 5600, f'E = ({m1:.2f} ± {errors1[0]:.2f}) * Chn + {c1:.2f} ± {errors1[1]:.2f}', fontsize=10)
plt.savefig(IMAGE_PATH/'TT_21_Chn1_calib.png')
plt.show()

print(f'Calibração_1: E = ({m1:.2f} ± {errors1[0]:.2f}) * Chn + {c1:.2f} ± {errors1[1]:.2f}')
df._append({'m': m1, 'c': c1, 'error_m': errors1[0], 'error_c': errors1[1]}, ignore_index=True)
# Calibração (channel 2)
params2, params_cov2 = curve_fit(linear, x2, energy)
errors2 = np.sqrt(np.diag(params_cov2))
m2 = params2[0]
c2 = params2[1]

plt.figure()
plt.plot(x2, energy, 'o', label='data')
plt.plot(x2, linear(x2, *params2), label='fit')
plt.title('Calibration channel 2')
plt.legend()
plt.xlabel('Channel')
plt.ylabel('Energy (keV)')
plt.text(565, 5600, f'E = ({m2:.2f} ± {errors2[0]:.2f}* Chn + {c2:.2f}) ± {errors2[1]:.2f}', fontsize=10)
plt.savefig(IMAGE_PATH/'TT_21_Chn2_calib.png')
plt.show()

print(f'Calibração_2: E = ({m2:.2f} ± {errors2[0]:.2f}) * Chn + {c2:.2f} ± {errors2[1]:.2f}')
df._append({'m': m2, 'c': c2, 'error_m': errors2[0], 'error_c': errors2[1]}, ignore_index=True)
df.to_csv(CALIB_FILE, index=False)

