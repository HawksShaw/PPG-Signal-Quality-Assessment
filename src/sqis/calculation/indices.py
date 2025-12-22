import numpy as np
from scipy import signal, stats

class SQIcalc:
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate

    def get_all_sqi(self, ppg_window, peaks):

        #1. Morphological consistency
        if len(peaks) > 1:
            template_correlation = self.get_template_correlation(ppg_window, peaks)
        else:
            template_correlation = 0.0

        #2. Time domain statistics
        skewness = self.get_skewness(ppg_window)
        kurtosis = self.get_kurtosis(ppg_window)
        hjorth   = self.get_hjorth(ppg_window)

        #3. Spectral descriptors
        entropy  = self.get_spectral_entropy(ppg_window)
        snr      = self.get_spectral_snr(ppg_window)

        return {
            "template_correlation": template_correlation,
            "skewness"            : skewness,
            "kurtosis"            : kurtosis,
            "hjorth_activity"     : hjorth['activity'],
            "hjorth_mobility"     : hjorth['mobility'],
            "hjorth_complexity"   : hjorth['complexity'],
            "spectral_entropy"    : entropy,
            "spectral_snr"        : snr
        }

    # --- Morphological Consistency - Template Correlation ---

    def get_template_correlation(self, ppg_window, peaks):
        
        sample_beats = []
        sampling_rate = int(self.fs)

        for i in range(len(peaks) - 1):
            start = peaks[i]
            end   = peaks[i+1]
            beat  = ppg_window[start:end]
            if len(beat) < 0.24*sampling_rate or len(beat) > 1.776*sampling_rate:
                continue
        
            resampled_beat = signal.resample(beat, sampling_rate)
            sample_beats.append(resampled_beat)

        if not sample_beats:
            return 0.0

        beats_array = np.array(sample_beats)
        beats_template = np.mean(beats_array, axis=0)
        beats_correlations = []
        for beat in beats_array:
            corr, _ = stats.pearsonr(beat, beats_template)
            beats_correlations.append(corr)
        return np.mean(beats_correlations)

    # --- Time Domain Statistics ---

    def get_skewness(self, ppg_window):
        return stats.skew(ppg_window)

    def get_kurtosis(self, ppg_window):
        return stats.kurtosis(ppg_window)

    def get_hjorth(self, ppg_window):
        window_prime = np.diff(ppg_window)
        window_bis   = np.diff(window_prime)

        variance_window = np.var(ppg_window)
        variance_prime  = np.var(window_prime)
        variance_bis    = np.var(window_bis)

        activity = variance_window
        mobility = np.sqrt(variance_prime/variance_window) if variance_window > 0 else 0
        complexity = np.sqrt(variance_bis/variance_prime)/mobility if mobility > 0 else 0

        return {
            "activity"   : activity,
            "mobility"   : mobility,
            "complexity" : complexity
        }

    # --- Frequency Domain Statistics ---
    def get_spectral_snr(self, ppg_window):

        freqs, psd = signal.welch(ppg_window, self.fs, nperseg=len(ppg_window))

        cardiac_band  = (freqs >= 0.5) & (freqs <= 3.7)
        cardiac_power = np.sum(psd[cardiac_band])
        total_power   = np.sum(psd)

        if total_power == 0:
            return 0.0

        return cardiac_power/total_power

    def get_spectral_entropy(self, ppg_window):

        freqs, psd = signal.welch(ppg_window, self.fs, nperseg=len(ppg_window))

        psd_normalised = psd/np.sum(psd)

        epsilon = 1e-14
        entropy = -np.sum(psd_normalised*np.log2(psd_normalised + epsilon))

        return entropy/np.log2(len(psd))
