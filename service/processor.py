import numpy as np
from sqis.calculation.fiducials import getFiducials
from sqis.calculation.indices import SQIcalc
from sqis.calculation.decision_policies import Decision
from sqis.calculation.imu import IMUDetector

class QualityChecker:
    def __init__(self):
        pass
    
    def get_indices(self, fs):
        return {
            "fids" : getFiducials(sampling_rate=fs),
            "sqis" : SQIcalc(sampling_rate=fs),
            "imu"  : IMUDetector(sampling_rate=fs, motion_threshold=0.2)
        }

    def z_score(self, signal):
        if np.std(signal) < 1e-6:
            return signal
        else:
            return (signal-np.mean(signal))/np.std(signal)
            
    def window_processing(self, ppg_ir:list, acc_x:list, acc_y:list, acc_z:list, fs:float):
        
        tools = self.get_indices(fs)

        signal_ir = self.z_score(np.array(ppg_ir))

        a_x = np.array(acc_x)
        a_y = np.array(acc_y)
        a_z = np.array(acc_z)

        is_artifact, imu_metrics = tools['imu'].check_motion(a_x, a_y, a_z)
        fids_dict = tools['fids'].extract_fiducials(signal_ir)
        peaks = fids_dict["systolic_peaks"]

        if len(peaks) >= 3 and not is_artifact:
            sqi_metrics = tools['sqis'].get_all_sqi(signal_ir, peaks)
            sqi_metrics.update(imu_metrics)
        else:
            sqi_metrics = imu_metrics

        policy = Decision()
        report = policy.decide(sqi_metrics=sqi_metrics, motion_flagged=is_artifact, num_peaks=len(peaks))

        return {
            "status": report.status.value,
            "confidence": float(report.confidence),
            "reasons": report.reasons,
            "metrics": report.metrics,
            "metadata": {
                "n_peaks": int(len(peaks)),
                "motion_detected": bool(is_artifact)
            }
        }