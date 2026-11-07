# Feature tipin_early_response_ratio (lens: longitudinal_powertrain) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: For the same clean tip-ins, the acceleration gain reached 0.4-0.6 s after the pedal flag divided by the peak gain within 2.5 s. A motor delivers its torque step almost at once (ratio near 1, limited only by the driver's pedal ramp), whereas turbo spool, converter lock-up and gear selection make an ICE build acceleration progressively (low ratio). Larger mass and larger turbocharged engines slow the build-up further.
# Hypothesis: BEV > ICE; SMALL > MID > LARGE
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled
# Training-sample statistics: finite 0.29, AUROC BEV-vs-ICE 0.536, Spearman(footprint) -0.200
# Audit: keep_with_caveat - Same clean tip-in events, dimensionless so it cancels pedal demand amplitude; physics sound (early torque delivery). Caveats: driver pedal ramp rate and platform pedal-map tuning (LARGE-BEV group 0.14 vs 0.41-0.45 drives the within-BEV size rho -0.33), gas-flag threshold per port; weak BEV/ICE separation (0.54). Spearman -0.68 with the latency feature, not a duplicate.
import numpy as np

def feat_tipin_early_response_ratio(w):
    """Median fraction of the 2.5-s peak acceleration gain already delivered 0.4-0.6 s after a tip-in (>= 1 event)."""
    v = np.asarray(w["v_mps"], dtype=np.float64)
    a = np.asarray(w["a_mps2"], dtype=np.float64).copy()
    a[np.abs(a) > 8.0] = np.nan
    gi = np.nan_to_num(np.asarray(w["gas_pressed"], dtype=np.float64), nan=0.0) > 0.5
    bi = np.nan_to_num(np.asarray(w["brake_pressed"], dtype=np.float64), nan=1.0) > 0.5
    di = np.nan_to_num(np.asarray(w["das_enabled"], dtype=np.float64), nan=1.0) > 0.5
    n = a.shape[0]
    vals = []
    for i in range(10, n - 25):
        if not (gi[i] and not gi[i - 1]):
            continue
        if gi[i - 10:i].any() or not gi[i:i + 10].all():
            continue
        if bi[i - 5:i + 15].any() or di[i - 10:i + 25].any():
            continue
        if np.nanmin(v[i - 5:i + 25]) < -0.1:
            continue
        pre = a[i - 5:i]
        if np.isnan(pre).sum() > 2:
            continue
        seg = a[i:i + 25] - np.nanmedian(pre)
        if np.isnan(seg).sum() > 5:
            continue
        pk = np.nanmax(seg)
        if not np.isfinite(pk) or pk < 0.6:
            continue
        early = np.nanmean(seg[4:7])
        if not np.isfinite(early):
            continue
        vals.append(min(max(early / pk, -0.5), 1.5))
    if len(vals) < 1:
        return float("nan")
    return float(np.median(vals))
