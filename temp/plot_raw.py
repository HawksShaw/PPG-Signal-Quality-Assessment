from patient_1load import *
import matplotlib.pyplot as plt
from signal_fourier import positive_mag, frequency_domain, inverted_fourier

plt.figure()
plt.subplot(1,2,1)
plt.plot(frequency_domain, positive_mag)
plt.xlabel('frequency [Hz]')
plt.ylabel('PPG [au]')

plt.subplot(1,2,2)
plt.plot(windowed_time, inverted_fourier)
plt.show()

