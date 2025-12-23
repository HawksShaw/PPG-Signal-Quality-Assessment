from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

file = loadmat('data/raw/Sample_PPG_MAT_125Hz.mat', struct_as_record = False, squeeze_me = True)

fs = file['Fs']
data = file['Data']

n = len(data)
delta_t = 1/fs

time = np.linspace(0, (n-1)*delta_t, n)

data_window = data[:fs*8]
time_window = time[:fs*8]

plt.figure()
plt.plot(time_window, data_window)
plt.show()