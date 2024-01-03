import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from reader import reader

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

# TT_20
TT20_Chn0 = reader(TT_20_PATH / 'CH0@N6781_21198_Espectrum_TT_20_20231207_141538.n42', 0)
TT20_Chn1 = reader(TT_20_PATH /'CH1@N6781_21198_Espectrum_TT_20_20231207_141538.n42', 1)
TT20_Chn2 = reader(TT_20_PATH /'CH2@N6781_21198_Espectrum_TT_20_20231207_141538.n42', 2)

# TT_21
TT21_Chn0 = reader(TT_21_PATH / 'CH0@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 0)
TT21_Chn1 = reader(TT_21_PATH / 'CH1@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 1)
TT21_Chn2 = reader(TT_21_PATH / 'CH2@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 2)

# TT_22
TT22_Chn0 = reader(TT_22_PATH / 'CH0@N6781_21198_Espectrum_TT_22_20231207_144117.n42', 0)
TT22_Chn1 = reader(TT_22_PATH / 'CH1@N6781_21198_Espectrum_TT_22_20231207_144117.n42', 1)
TT22_Chn2 = reader(TT_22_PATH / 'CH2@N6781_21198_Espectrum_TT_22_20231207_144117.n42', 2)

# TT_23
TT23_Chn0 = reader(TT_23_PATH / 'CH0@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 0)
TT23_Chn1 = reader(TT_23_PATH / 'CH1@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 1)
TT23_Chn2 = reader(TT_23_PATH / 'CH2@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 2)

# TT_24
TT24_Chn0 = reader(TT_24_PATH / 'CH0@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 0)
TT24_Chn1 = reader(TT_24_PATH / 'CH1@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 1)
TT24_Chn2 = reader(TT_24_PATH / 'CH2@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 2)

# TT_25
TT25_Chn0 = reader(TT_25_PATH / 'CH0@N6781_21198_Espectrum_TT_25_20231207_150004.n42', 0)
TT25_Chn1 = reader(TT_25_PATH / 'CH1@N6781_21198_Espectrum_TT_25_20231207_150004.n42', 1)
TT25_Chn2 = reader(TT_25_PATH / 'CH2@N6781_21198_Espectrum_TT_25_20231207_150004.n42', 2)

# TT_26
TT26_Chn0 = reader(TT_26_PATH / 'CH0@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 0)
TT26_Chn1 = reader(TT_26_PATH / 'CH1@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 1)
TT26_Chn2 = reader(TT_26_PATH / 'CH2@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 2)

# TT_27
TT27_Chn0 = reader(TT_27_PATH / 'CH0@N6781_21198_Espectrum_TT_27_20231207_160722.n42', 0)
TT27_Chn1 = reader(TT_27_PATH / 'CH1@N6781_21198_Espectrum_TT_27_20231207_160722.n42', 1)
TT27_Chn2 = reader(TT_27_PATH / 'CH2@N6781_21198_Espectrum_TT_27_20231207_160722.n42', 2)


# Plotting the data

# TT_20 -> 2048 bins, IGNORAR
plt.figure()
plt.plot(TT20_Chn0, label='Chnannel 0')
plt.plot(TT20_Chn1, label='Channel 1')
plt.plot(TT20_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_20')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_20.png')    
plt.show()

# TT_21 -> 1024 bins, 10min de aquisição
plt.figure()
plt.plot(TT21_Chn0, label='Chnannel 0')
plt.plot(TT21_Chn1, label='Channel 1')
plt.plot(TT21_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_21')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_21.png')
plt.show()

# TT_22 -> Error
plt.figure()
plt.plot(TT22_Chn0, label='Chnannel 0')
plt.plot(TT22_Chn1, label='Channel 1')
plt.plot(TT22_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_22')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_22.png')
plt.show()

# TT_23 -> 1024 bins, 2mC (LiF)
plt.figure()
plt.plot(TT23_Chn0, label='Chnannel 0')
plt.plot(TT23_Chn1, label='Channel 1')
plt.plot(TT23_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_23')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_23.png')
plt.show()

# TT_24 -> 1024 bins, 2mC (LiAlO2)
plt.figure()
plt.plot(TT24_Chn0, label='Chnannel 0')
plt.plot(TT24_Chn1, label='Channel 1')
plt.plot(TT24_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_24')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_24.png')
plt.show()

# TT_25 -> 1024 bins, 2mC (Li Implantado em Al)
plt.figure()
plt.plot(TT25_Chn0, label='Chnannel 0')
plt.plot(TT25_Chn1, label='Channel 1')
plt.plot(TT25_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_25')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_25.png')
plt.show()

# TT_26
plt.figure()
plt.plot(TT26_Chn0, label='Chnnanel 0')
plt.plot(TT26_Chn1, label='Channel 1')
plt.plot(TT26_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_26')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_26.png')
plt.show()

# TT_27
plt.figure()
plt.plot(TT27_Chn0, label='Chnannel 0')
plt.plot(TT27_Chn1, label='Channel 1')
plt.plot(TT27_Chn2, label='Channel 2')
plt.legend()
plt.title('TT_27')
plt.yscale('log')
plt.xlabel('Channel')
plt.ylabel('Counts')
plt.savefig(IMAGE_PATH/'TT_27.png')
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
plt.savefig(IMAGE_PATH/'TT_21_Chn0.png')
plt.show()









