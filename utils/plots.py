import numpy as np
from .calibration import compute_calibration


def get_fig_records(info, N_groups=5, **metadata):
    records = []
    calibrated_probs = info["probs"]
    labels = info["labels"]
    atypicality = info["input_atypicality"].flatten()
    for calibration_method in calibrated_probs:
        probs_calib = calibrated_probs[calibration_method]
        preds_calib = np.argmax(probs_calib, axis=1)
        quantiles = np.linspace(0, 1, N_groups)
        for q_lower, q_higher in zip(quantiles[:-1], quantiles[1:]):
            vs = np.quantile(atypicality, q=[q_lower, q_higher])
            # Control for the start
            if q_lower == 0:
                vs[0] = -np.inf
            mask = (atypicality <= vs[1]) & (atypicality > vs[0])
            group_probs = probs_calib[mask]
            group_lbls = labels[mask]
            group_atypicality = atypicality[mask]
            calib_metrics = compute_calibration(
                group_lbls,
                group_probs
            )

            record = {
                "ECE": calib_metrics["expected_calibration_error"],
                "RMSE": calib_metrics["rms_calibration_error"],
                "quantile": q_higher,
                "size": mask.sum(),
                "Accuracy": (np.argmax(group_probs, axis=1) == group_lbls).mean(),
                "Recalibration": calibration_method,
                "MeanAtypicality": group_atypicality.mean(),
                **metadata
            }
            records.append(record)
    return records
