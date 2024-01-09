import numpy as np
import pandas as pd
from pathlib import Path

# Path for the caliration csv 
CALIB_FILE = Path('data/calibration.csv')


# Load the calibration data
calibration_data = pd.read_csv(CALIB_FILE)

# Define Gaussian function
def gaussian(x, a, x0, sigma, c=0):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c

# -----------------------------------------------------------------------------
# Calibration function for channel 0
m0 = calibration_data['m'].to_numpy()[0]
b0 = calibration_data['c'].to_numpy()[0]
error_m0 = calibration_data['error_m'].to_numpy()[0]
error_b0 = calibration_data['error_c'].to_numpy()[0]

def calibration_Ch0(x):
    return m0 * x + b0

def error_Ch0(bins, error_bins):
    '''Error propagation for the calibration function of channel 0.
    The error is calculated using the errorpropagation website,
    absolute error.'''
    return abs(m0) * error_bins + abs(bins) * error_m0 + error_b0

# Calibration function for channel 1
m1 = calibration_data['m'].to_numpy()[1]
b1 = calibration_data['c'].to_numpy()[1]
error_m1 = calibration_data['error_m'].to_numpy()[1]
error_b1 = calibration_data['error_c'].to_numpy()[1]

def calibration_Ch1(x):
    return m1 * x + b1

def error_Ch1(bins, error_bins):
    return abs(m1) * error_bins + abs(bins) * error_m1 + error_b1

# Calibration function for channel 2
m2 = calibration_data['m'].to_numpy()[2]
b2 = calibration_data['c'].to_numpy()[2]
error_m2 = calibration_data['error_m'].to_numpy()[2]
error_b2 = calibration_data['error_c'].to_numpy()[2]

def calibration_Ch2(x):
    return m2 * x + b2

def error_Ch2(bins, error_bins):
    return abs(m2) * error_bins + abs(bins) * error_m2 + error_b2

