import numpy as np
from sqis.calculation.fiducials import getFiducials
from sqis.calculation.indices import SQIcalc
from sqis.calculation.decision_policies import Decision
from sqis.calculation.imu import IMUDetector

class QualityChecker:
    def __init__(self):
        self.current_fs = None
        self.fids = None
        self.sqis = None
        self.imu = None
        self.decision_policy = Decision()

    def is_initialized(self, fs):
        if self.current_fs != fs:
            print(f"[Service layer] Switching current sampling rate to match extracted sampling rate")
            self.current_fs = fs
            self.fids = getFiducials(sampling_rate=fs)
            self.sqis = SQIcalc(sampling_rate=fs)
            self.imu = IMUDetector(sampling_rate=fs, motion_threshold=0.2)
            
    def window_processing(self, ppg_ir:list, acc_x:list, acc_y:list, acc_z:list, fs:float):
        self.is_initialized(fs)

        signal_ir = np.array(ppg_ir)
        a_x = np.array(acc_x)
        a_y = np.array(acc_y)
        a_z = np.array(acc_z)

        is_artifact, imu_metrics = self.imu.check_motion(a_x, a_y, a_z)

        fids_dict = self.fids.extract_fiducials(signal_ir)
        peaks = fids_dict["systolic_peaks"]

        if len(peaks) >= 3 and not is_artifact:
            sqi_metrics = self.sqis.get_all_sqi(signal_ir, peaks)
            sqi_metrics.update(imu_metrics)
        else:
            sqi_metrics = imu_metrics

        report = self.decision_policy.decide(sqi_metrics=sqi_metrics, motion_flagged=is_artifact, num_peaks=len(peaks))

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