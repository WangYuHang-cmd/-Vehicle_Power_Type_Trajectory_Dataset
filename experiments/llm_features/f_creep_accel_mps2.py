# Feature creep_accel_mps2 (lens: longitudinal_powertrain) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Mean acceleration while rolling at 0.05-2.5 m/s with no pedal for >= 0.5 s, no automation and not at standstill. A torque-converter ICE creeps forward on idle torque (positive acceleration up to walking speed), while a BEV without creep or in one-pedal/hold mode has no idle torque and regenerates down to a stop (negative acceleration). BEVs with creep emulation sit near zero.
# Hypothesis: BEV < ICE
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled, standstill
# Training-sample statistics: finite 0.11, AUROC BEV-vs-ICE 0.166, Spearman(footprint) -0.080
# Audit: keep_with_caveat - Clean creep-vs-regen sign flip (BEV -0.46 vs ICE +0.08, HEV +0.29 creep-emulation, AUROC 0.17) with a 0.5-s settle requirement. Caveats: creep is a selectable BEV mode; aEgo between 0.05 and 0.3 m/s sits in port-specific wheel-speed dead zones; coverage 11.3% is just above the floor. Retained over its near-duplicate lowspeed_nopedal_accel (0.97).
import numpy as np

def feat_creep_accel_mps2(w):
    """Mean acceleration while rolling at 0.05-2.5 m/s with no pedal pressed for >= 0.5 s and no automation: torque-converter creep vs no-creep / regen-to-stop."""
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
    ok[5:] = (c[6:] - c[:-6]) == 6
    m = ok & (v > 0.05) & (v < 2.5) & np.isfinite(a)
    if m.sum() < 8:
        return float("nan")
    return float(np.mean(a[m]))
