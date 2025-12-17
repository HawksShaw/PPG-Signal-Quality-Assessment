from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

file = loadmat('data/raw/part_1.mat', struct_as_record = False, squeeze_me = True)

ppg_1 = file['p'][0][0]
abp_1 = file['p'][0][1]
ecg_1 = file['p'][0][2]

fs = 125
delta_t = 1/fs
N_1 = len(ppg_1)
time_1 = np.linspace(0, N_1*delta_t, N_1)

plt.figure()
plt.subplot(3,1,1)
plt.plot(time_1, ppg_1)
plt.title('PPG v time')

plt.subplot(3,1,2)
plt.plot(time_1, abp_1)
plt.title('ABP v time')

plt.subplot(3,1,3)
plt.plot(time_1, ecg_1)
plt.title('ECG v time')

plt.show()
