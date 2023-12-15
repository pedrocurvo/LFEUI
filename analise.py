import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

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

# Reading the data files

# TT_20
TT20_Chn0 = reader('data/TT_20/UNFILTERED/CH0@N6781_21198_Espectrum_TT_20_20231207_141538.n42', 0)
TT20_Chn1 = reader('data/TT_20/UNFILTERED/CH1@N6781_21198_Espectrum_TT_20_20231207_141538.n42', 1)
TT20_Chn2 = reader('data/TT_20/UNFILTERED/CH2@N6781_21198_Espectrum_TT_20_20231207_141538.n42', 2)

# TT_21
TT21_Chn0 = reader('data/TT_21/UNFILTERED/CH0@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 0)
TT21_Chn1 = reader('data/TT_21/UNFILTERED/CH1@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 1)
TT21_Chn2 = reader('data/TT_21/UNFILTERED/CH2@N6781_21198_Espectrum_TT_21_20231207_143401.n42', 2)

# TT_22
TT22_Chn0 = reader('data/TT_22/UNFILTERED/CH0@N6781_21198_Espectrum_TT_22_20231207_144117.n42', 0)
TT22_Chn1 = reader('data/TT_22/UNFILTERED/CH1@N6781_21198_Espectrum_TT_22_20231207_144117.n42', 1)
TT22_Chn2 = reader('data/TT_22/UNFILTERED/CH2@N6781_21198_Espectrum_TT_22_20231207_144117.n42', 2)

# TT_23
TT23_Chn0 = reader('data/TT_23/UNFILTERED/CH0@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 0)
TT23_Chn1 = reader('data/TT_23/UNFILTERED/CH1@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 1)
TT23_Chn2 = reader('data/TT_23/UNFILTERED/CH2@N6781_21198_Espectrum_TT_23_20231207_144752.n42', 2)

# TT_24
TT24_Chn0 = reader('data/TT_24/UNFILTERED/CH0@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 0)
TT24_Chn1 = reader('data/TT_24/UNFILTERED/CH1@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 1)
TT24_Chn2 = reader('data/TT_24/UNFILTERED/CH2@N6781_21198_Espectrum_TT_24_20231207_145454.n42', 2)

# TT_25
TT25_Chn0 = reader('data/TT_25/UNFILTERED/CH0@N6781_21198_Espectrum_TT_25_20231207_150004.n42', 0)
TT25_Chn1 = reader('data/TT_25/UNFILTERED/CH1@N6781_21198_Espectrum_TT_25_20231207_150004.n42', 1)
TT25_Chn2 = reader('data/TT_25/UNFILTERED/CH2@N6781_21198_Espectrum_TT_25_20231207_150004.n42', 2)

# TT_26
TT26_Chn0 = reader('data/TT_26/UNFILTERED/CH0@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 0)
TT26_Chn1 = reader('data/TT_26/UNFILTERED/CH1@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 1)
TT26_Chn2 = reader('data/TT_26/UNFILTERED/CH2@N6781_21198_Espectrum_TT_26_20231207_152550.n42', 2)

# TT_27
TT27_Chn0 = reader('data/TT_27/UNFILTERED/CH0@N6781_21198_Espectrum_TT_27_20231207_160722.n42', 0)
TT27_Chn1 = reader('data/TT_27/UNFILTERED/CH1@N6781_21198_Espectrum_TT_27_20231207_160722.n42', 1)
TT27_Chn2 = reader('data/TT_27/UNFILTERED/CH2@N6781_21198_Espectrum_TT_27_20231207_160722.n42', 2)


