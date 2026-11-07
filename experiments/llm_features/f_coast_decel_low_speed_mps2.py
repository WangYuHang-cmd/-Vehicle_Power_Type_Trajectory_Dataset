# Feature coast_decel_low_speed_mps2 (lens: longitudinal_powertrain) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Mean acceleration while rolling at 3-12 m/s with neither pedal pressed and no automation for at least 1 s. In this speed band an ICE automatic unlocks its torque converter and engine braking fades to near zero (creep torque even opposes drag), whereas an electric motor's regen torque is at its strongest (torque-limited rather than power-limited). Deceleration is drivetrain drag torque divided by mass, so heavier vehicles decelerate less for the same drivetrain.
# Hypothesis: BEV < ICE (more negative); LARGE > MID > SMALL (less negative for heavier vehicles)
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled, standstill
# Training-sample statistics: finite 0.14, AUROC BEV-vs-ICE 0.158, Spearman(footprint) 0.290
# Audit: keep_with_caveat - Sound (regen stays torque-limited where the torque converter unlocks), confirmed (BEV -1.11 vs ICE -0.25, AUROC 0.16); its size trend partly survives within ICE (rho 0.17), consistent with drag torque per mass. Caveats: regen setting, grade, and near-duplication with liftoff_decel_moving (0.87) and coast_c0_retard (-0.94).
import numpy as np

def feat_coast_decel_low_speed_mps2(w):
    """Mean acceleration while coasting at 3-12 m/s (no pedal/automation >= 1.0 s): regen keeps pulling where an ICE torque converter unlocks and engine braking fades."""
    v = np.asarray(w["v_mps"], dtype=np.float64)
    a = np.asarray(w["a_mps2"], dtype=np.float64).copy()
    a[np.abs(a) > 8.0] = np.nan
    gi = np.nan_to_num(np.asarray(w["gas_pressed"], dtype=np.float64), nan=0.0) > 0.5
    bi = np.nan_to_num(np.asarray(w["brake_pressed"], dtype=np.float64), nan=1.0) > 0.5
    di = np.nan_to_num(np.asarray(w["das_enabled"], dtype=np.float64), nan=1.0) > 0.5
    st = np.nan_to_num(np.asarray(w["standstill"], dtype=np.float64), nan=1.0) > 0.5
    free = (~gi) & (~bi) & (~di) & (~st)
    n = a.shape[0]
    c = np.concatenate(([0], np.cumsum(free.astype(np.int64))))
    ok = np.zeros(n, dtype=bool)
    ok[10:] = (c[11:] - c[:-11]) == 11
    m = ok & (v >= 3.0) & (v < 12.0) & np.isfinite(a)
    if m.sum() < 10:
        return float("nan")
    return float(np.mean(a[m]))
