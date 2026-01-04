import numpy as np
import heartpy as hp
from scipy import signal
from peaks_elgendi import peaks_elgendi

class getFiducials:
    def __init__(self, sampling_rate=128):
        self.fs = sampling_rate
        self.min_distance = int(0.27*self.fs)


    def peaks_heartpy(self, signal_window):
        try:
            working_data, measures = hp.process(signal, sample_rate=fs, clean_rr=True)
            peaks = np.array(working_data['peaklist'])
            return peaks
        except Exception as e:
            print(f"Error while analyzing systolic peaks: {e}")
            return np.array([])
        

    def peaks_elgendi(self, signal_window):
        """
        Ref: Elgendi M. et al., 'Systolic Peak Detection in Acceleration Photoplethysmograms...'
        """
        #Square the signal
        signal_sq = np.power(signal, 2)

        #Define moving average window sizes
        fast_window = int(0.11 * fs) | 1 #fast window for expected ~0.11s-wide systolic peak
        slow_window = int(0.60 * fs) | 1 #slow window for expected ~0.60s-long heartbeat

        ma_peak = pd.Series(signal_sq).rolling(window=fast_window, center=True).mean().fillna(0).values
        ma_beat = pd.Series(signal_sq).rolling(window=slow_window, center=True).mean().fillna(0).values

        beta = 0.02 * np.mean(signal_sq)
        roi  = ma_peak > (ma_beat + beta)

        peaks = []
        peak_search = True

        start_region = 0
        in_region = False
        
        for i, is_active in enumerate(roi):
            if is_active and not in_region:
                in_region = True
                start_region = i
            elif not is_active and in_region:
                in_region = False
                end_region = i
                if end_region > start_region:
                    segment = signal_window[start_region:end_region]
                    local_max_idx = np.argmax(segment) + start_region
                    peaks.append(local_max_idx)

        peaks = np.array(peaks)
        if len(peaks) > 1:
            kept_indices = []
            last_peak = -self.min_distance

            for peak in peaks:
                if peak - last_peak > self.min_distance:
                    kept_indices.append(peak)
                    last_peak = peak
                else:
                    if signal_window[peak] > signal_window[kept_indices[-1]]:
                        kept_indices[-1] = peak
                        last_peak = peak
        return peaks

    def extract_fiducials(self, ppg_window):
        
        # --- ADJUST COMMENTS WHEN NEEDED ---
        peaks = self.peaks_elgendi(ppg_window)
        #peaks = self.peaks_heartpy(ppg_window)

        notches, diastolic_peaks = self.find_diastoles(ppg_window, peaks)
        pulse_onsets = self.find_onsets(ppg_window, peaks)


        return {
            "systolic_peaks" : peaks,
            "diastolic_peaks" : diastolic_peaks,
            "notches" : np.array(notches),
            "pulse_onsets" : pulse_onsets,
        }

    def find_diastoles(self, ppg_window, peaks):
        notches = []
        diastoles = []

        for i, peak in enumerate(peaks):
            search_start = peak+int(0.05*self.fs)
            search_end   = peak+int(0.55*self.fs)

            if search_start >= len(ppg_window):
                notches.append(None)
                diastoles.append(None)
                continue
            
            search_end = min(search_end, len(ppg_window))
            search_segment = ppg_window[search_start : search_end]

            if len(search_segment) < 3:
                notches.append(None)
                diastoles.append(None)
                continue

            local_minima_idx, _ = signal.find_peaks(-search_segment)
            local_maxima_idx, _ = signal.find_peaks(search_segment)

            if len(local_minima_idx) > 0 and len(local_maxima_idx) > 0:
                notch_idx = search_start + local_minima_idx[0]
                diastole_idx_valid = [x for x in local_maxima_idx if x > local_minima_idx[0]]

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