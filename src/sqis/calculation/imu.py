import numpy as np

class IMUDetector:
    def __init__(self, sampling_rate=128, motion_threshold=0.2):
        self.fs = sampling_rate
        self.threshold = motion_threshold

    def check_motion(self, acc_x, acc_y, acc_z):
        
        #1. Get magnitude
        magnitute = np.sqrt(acc_x**2 + acc_y**2, acc_z**2)

        #2. Get dynamic component
        motion_std = np.std(magnitude)

        #3. Calculate jerk
        jerk = np.diff(magnitude)
        avg_jerk = np.mean(abs(jerk)) if len(jerk) > 0 else 0.0

        is_artifact = motion_std > self.threshold

        return is_artifact, {
            "motion_energy" : motion_std,
            "average_jerk"  : avg_jerk,
            "max_magnitude" : np.max(magnitude)
        }


