from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

class SignalStatus(Enum):
    GOOD       = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    BAD        = "BAD"

@dataclass
class QualityReport:
    status: SignalStatus
    confidence: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str] = field(default_factory=dict)

class Decision:
    def __init__(self):
        self.thresholds = {
            "activity_min" : 0.0001,
            "min_peaks": 3,

            "template_corr_min": 0.65,
            "snr_min"          : 0.3,
            "skew_min"         : 0.1
        }

        self.weights = {
            "template_corr": 0.6,
            "snr"          : 0.3,
            "skew"         : 0.1
        }

    def decide(self, sqi_metrics: dict, motion_flagged: bool, num_peaks: int) -> QualityReport:
        
        #Hardgating
        if motion_flagged:
            return QualityReport(
                status = SignalStatus.BAD,
                confidence = 0.0,
                reasons = ["Critical Motion Exceeded"],
                metrics = sqi_metrics
            )

        if num_peaks < self.thresholds["min_peaks"]:
            return QualityReport(
                status = SignalStatus.BAD,
                confidence = 0.0,
                reasons = [f"Insufficient number of peaks (Found {num_peaks})"],
                metrics = sqi_metrics
            )

        if sqi_metrics.get("hjorth_activity", 0) < self.thresholds["activity_min"]:
            return QualityReport(
                status = SignalStatus.BAD,
                confidence = 0.0,
                reasons = [f"Sensor disconnected or flat"],
                metrics = sqi_metrics
            )
        
        #Softgating
        reasons = []

        template_corr = sqi_metrics.get("template_correlation", 0)
        if template_corr < self.thresholds["template_corr_min"]:
            reasons.append(f"Poor Morphology (Corr: {template_corr:.2f})")

        snr = sqi_metrics.get("spectral_snr", 0)
        if snr < self.thresholds["snr_min"]:
            reasons.append(f"High Signal-to-Noise Ratio (snr: {snr:.2f})")
        
        skewness = sqi_metrics.get("skewness", 0)
        if skewness < self.thresholds["skew_min"]:
            reasons.append(f"Bad peak distribution (Skewness: {skewness:.2f})")

        #Scoring
        score = template_corr*self.weights["template_corr"] + snr*self.weights["snr"] + skewness*self.weights["skew"]
        
        total_weight = sum(self.weights.values())
        total_score = score / total_weight if total_weight > 0 else 0

        if len(reasons) == 0:
            final_verdict = SignalStatus.GOOD
        elif len(reasons) <= 1 and total_score > 0.5:
            final_verdict = SignalStatus.ACCEPTABLE
        else:
            final_verdict = SignalStatus.BAD
        
        return QualityReport(
            status=final_verdict,
            confidence=round(total_score, 2),
            reasons=reasons,
            metrics=sqi_metrics
        )
        