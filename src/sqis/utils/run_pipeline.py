import pandas as pd
import numpy as np
from sqis.calculation.hr_quality import hr_quality
from sqis.calculation.fiducials import getFiducials
from sqis.calculation.indices import SQIcalc
from sqis.calculation.imu import IMUDetector
from sqis.calculation.decision_policies import SignalStatus, Decision
from sqis.preprocessing.wildppg_init import wildppg_stream
import time

def run_pipeline(data_path):
    stream = wildppg_stream(data_path, window_seconds=8, preprocess=True)

    fids = None
    imu  = None
    sqis = None
    decision_policy = Decision()
    results = []

    current_fs = 0

    for i, window in enumerate(stream):
        try:
            # #!!! PLACEHOLDER !!!
            # print(f"Current iteration: {i}")
            # #!!! PLACEHOLDER !!!

            signal_ir = window['ppg_signal']['ir']
            signal_acc = window['accel']
            fs = window['metadata']['sampling_rate']

            if current_fs != fs:
                print(f"Changing the current sampling rate to fit extracted sampling rate. Current FS: {current_fs}, Extracted FS: {fs}")
                current_fs = fs
                fids = getFiducials(sampling_rate=fs)
                imu  = IMUDetector(sampling_rate=fs, motion_threshold=0.2)
                sqis = SQIcalc(sampling_rate=fs)

            is_artifact, imu_metrics = imu.check_motion(signal_acc['x'], signal_acc['y'], signal_acc['z'])
            fids_dict = fids.extract_fiducials(signal_ir)
            peaks = fids_dict["systolic_peaks"]

            if len(peaks) >= 3 and not is_artifact:
                sqi_metrics = sqis.get_all_sqi(signal_ir, peaks)
                sqi_metrics.update(imu_metrics)
            else:
                sqi_metrics = imu_metrics

            report = decision_policy.decide(sqi_metrics=sqi_metrics, motion_flagged=is_artifact, num_peaks=len(peaks))

            gt_label = "Unknown"
            hr_error = np.nan
            if 'ecg' in window:
                gt_label, hr_error = hr_quality(peaks, window['ecg'], fs)

            logging = {
                'subject' : window['metadata']['subject_id'],
                'window_idx' : i,
                
                'n_peaks' : len(peaks),
                'motion_flag' : is_artifact,
                'sqi_skew' : sqi_metrics.get('skewness', 0),
                'sqi_corr' : sqi_metrics.get('template_correlation', 0),

                'sys_status' : report.status.value,
                'sys_confidence' : report.confidence,
                'sys_reasons' : "|".join(report.reasons),

                'gt_label' : gt_label,
                'hr_error' : hr_error
            }
            results.append(logging)

            if i % 50 == 0:
                print(f"processed {i} windows with following scoring: {report.status.value}")
        
        except Exception as e:
            print(f"Error: {e}")
            continue

    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = 'benchmark_results_{timestamp}.csv'
    df.to_csv(output_path, index=False)
    print(f"Benchmark complete for {len(df)} windows. Saving results to {output_path}.")
    return df
