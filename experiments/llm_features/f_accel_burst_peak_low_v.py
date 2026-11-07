# Feature accel_burst_peak_low_v (lens: energy_speed_profile) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Low-speed acceleration capability actually used: median over sustained human-driven acceleration bursts (a > 0.4 m/s^2 for at least 0.8 s, automation off) that begin below 8 m/s of the burst's peak acceleration. Defined on acceleration events rather than on the accelerator flag because in one-pedal BEVs the accelerator is pressed almost continuously (including while holding speed), which would dilute any pedal-conditioned percentile. In the torque-limited region peak a ~ T_max*ratio/(r*m): electric motors deliver peak torque from zero speed and BEVs have high low-speed specific force, while large family vehicles have lower specific force.
# Hypothesis: BEV > ICE; LARGE < SMALL. Modest effect sizes expected because drivers rarely use full torque.
# Channels: v_mps, a_mps2, das_enabled
# Training-sample statistics: finite 0.20, AUROC BEV-vs-ICE 0.520, Spearman(footprint) 0.065
# Audit: keep_with_caveat - Event-based (not pedal-flag conditioned, so immune to one-pedal dilution) and physically motivated, but realised peak acceleration is dominated by driver throttle habit; weak (AUROC 0.52, BEV 1.61 vs ICE 1.48) and no size trend. Keep as a modest specific-force cue with the driver-habit caveat.
import numpy as np

def feat_accel_burst_peak_low_v(w):
    v = w["v_mps"]; a = w["a_mps2"]; das = w["das_enabled"]
    m = np.isfinite(a) & np.isfinite(v) & (a > 0.4) & (das < 0.5)
    mm = np.concatenate(([0], m.astype(np.int8), [0]))
    d = np.diff(mm)
    starts = np.flatnonzero(d == 1); ends = np.flatnonzero(d == -1)
    peaks = []
    for s, e in zip(starts, ends):
        if e - s >= 8 and v[s] < 8.0:                   # >= 0.8 s burst starting below 8 m/s
            p = float(np.nanmax(a[s:e]))
            if np.isfinite(p):
                peaks.append(p)
    if len(peaks) < 2:
        return float("nan")
    return float(np.median(peaks))
