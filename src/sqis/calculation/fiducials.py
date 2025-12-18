from .systolic import systolic_peaks
from .diastolic import diastolic_peaks
from .dicrotic import dicrotic_notches
from .onsets import pulse_onsets
import matplotlib.pyplot as plt

def fiducials(signal, fs, prominence=None, height=None, pulse_fraction_percentage=0.3):

    signal_systoles, signal_props = systolic_peaks(signal=signal, fs = fs, prominence=prominence, height=None)

    signal_diastoles = diastolic_peaks(signal=signal, systolic_index=signal_systoles, fs=fs, prominence=prominence)

    signal_notches = dicrotic_notches(signal=signal, systolic_index=signal_systoles, diastolic_index=signal_diastoles)

    signal_onsets = pulse_onsets(signal=signal, fs=fs, systolic_index=signal_systoles, pulse_fraction_percentage=pulse_fraction_percentage)

    return signal_systoles, signal_diastoles, signal_notches, signal_onsets

def plot_fiducials(signal, systoles, diastoles, notches, onsets, time, legend=False):

    plt.figure()
    plt.plot(time, signal)
    plt.plot(time[systoles], signal[systoles], 'ro', label='Systolic peaks')
    plt.plot(time[diastoles], signal[diastoles], 'bo', label='Diastolic peaks')
    plt.plot(time[notches], signal[notches], 'go', label='Dicrotic notches')
    plt.plot(time[onsets], signal[onsets], 'co', label='Pulse onsets')

    if legend == True:
        plt.legend()
    
    plt.show()