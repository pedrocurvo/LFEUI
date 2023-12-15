import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import scipy as scipy

# Function to read the data from the file
def reader (name, Chn):
    # Parse the XML data from the file
    tree = ET.parse(name)
    root = tree.getroot()
    # Access spectrum data
    spectrum_data_str = root.find(f".//Spectrum[@id='RadMeasurement-{Chn}_Spectrum-{Chn}']/ChannelData").text
    # Convert the spectrum data string to a vector of data
    spectrum_data_array = np.array([float(value) for value in spectrum_data_str.split()])

    return spectrum_data_array

# TT_21
TT21_Chn0 = reader('data/TT_21/UNFILTERED/CH0@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 0)
TT21_Chn1 = reader('data/TT_21/UNFILTERED/CH1@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 1)
TT21_Chn2 = reader('data/TT_21/UNFILTERED/CH2@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 2)

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
plt.savefig('TT_21.png')
plt.show()

# Calibration

# For the calibration we will use the data from the TT_21 file

plt.figure()
plt.plot(TT21_Chn0, label='Chnannel 0')
plt.legend()
plt.title('TT_21')
plt.xlabel('Channel')
plt.yscale('log')
plt.ylabel('Counts')
plt.xlim(500, 700)
plt.ylim(1, 1e3)
plt.savefig('TT_21_Chn0.png')
plt.show()

# Gaussian fit for the peaks (soma 3 de gaussianas)
def gaussian(x, a1, x01, sigma1, a2, x02, sigma2, a3, x03, sigma3, c):
    return a1*np.exp(-(x-x01)**2/(2*sigma1**2)) + a2*np.exp(-(x-x02)**2/(2*sigma2**2)) + a3*np.exp(-(x-x03)**2/(2*sigma3**2)) + c

# Peaks
x = np.array(range(500, 701))
print(x)
y = TT21_Chn0[500:700]
print(y)
scipy.optimize.curve_fit(gaussian, x,y)

# valores de energia para os 3 picos
energy = np.array([5156.59, 5485.56, 5804.82])