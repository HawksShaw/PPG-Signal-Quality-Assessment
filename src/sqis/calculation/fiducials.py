import numpy as np
from scipy import signal

class getFiducials:
    def __init__(self, sampling_rate=128):
        self.fs = sampling_rate
        self.peak_height = 0.5
        self.min_distance = int(0.27*self.fs)

    def extract_fiducials(self, ppg_window):
        current_threshold = 0.6*peak_height

        peaks, props = signal.find_peaks(ppg_window, distance=self.min_distance, height=current_threshold)

        if len(props['peak heights']) > 0:
            avg_height = np.mean(props['peak_heights'])
            self.peak_height = 0.9*self.peak_height + 0.1*avg_height

            notches, diastolic_peaks = self.find_diastoles(ppg_window, peaks)
            pulse_onsets = self.find_onsets(ppg_window, peaks)


            return {
                "systolic_peaks" : peaks,
                "diastolic_peaks" : diastolic_peaks,
                "notches" : notches,
                "pulse_onsets" : pulse_onsets,
                "mean_height" : self.peak_height
            }

    def find_diastoles(self, ppg_window, peaks):
        notches = []
        diastoles = []

        for i, peak in enumerate(peaks):
            search_start = peak+int(0.05*self.fs)
            search_end   = peak+int(0.40*self.fs)

            if search_start >= len(window):
                notches.append(None)
                diastoles.append(None)
                continue
            
            search_end = np.min(search_end, len(window))
            search_segment = ppg_window[search_start : search_end]

            if len(search_segment) < 3:
                notches.append(None)
                diastoles.append(None)
                continue

            local_minima_idx, _ = signal.find_peaks(-search_segment)
            local_maxima_idx, _ = signal.find_peaks(search_segment)

            if len(local_minima_idx) > 0 and len(local_maxima_idx) > 0:
                notch_idx = search_start + local_minima_idx[0]
                diastole_idx_valid = [maximum for x in local_maxima_idx if x > local_minima_idx[0]]

                if diastole_idx_valid:
                    diastole_idx = search_start + diastole_idx_valid[0]
                    notches.append(notch_idx)
                    diastoles.append(diastole_idx)
                else:
                    notches.append(notch_idx)
                    diastoles.append(None)
            else:
                notches.append(None)
                diastoles.append(None)

        return notches, diastoles

    def find_onsets(self, ppg_window, peaks):
        onsets = []
        for i, peak in enumerate(peaks):
            previous_index = peaks[i-1] if i > 0 else 0
            search_segment = ppg_window[previous_index:peak]

            if len(search_segment) == 0:
                onsets.append(None)
                continue

            local_minima_idx = np.argmin(search_segment)
            onsets.append(previous_index + local_minima_idx)
        return onsets