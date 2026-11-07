# Feature liftoff_decel_moving (lens: energy_speed_profile) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Deceleration of the body when the driver has fully released both pedals while moving. In an ICE car this is engine braking in gear plus rolling/aero drag (about -0.15 to -0.4 m/s^2). In a BEV the lift-off torque is regenerative braking, typically 0.05-0.2 g and strongest when the pedal is fully released, so the coast deceleration is 2-5 times larger. Automation is excluded per sample because an ACC/openpilot controller can brake without pedals.
# Hypothesis: BEV << ICE (more negative; BEV around -0.7 m/s^2 vs ICE around -0.25). No strong size dependence expected (drag and engine braking scale roughly with mass).
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled, standstill
# Training-sample statistics: finite 0.32, AUROC BEV-vs-ICE 0.295, Spearman(footprint) -0.036
# Audit: keep_with_caveat - Clean and strong (BEV -0.70 vs ICE -0.23 m/s^2, AUROC 0.30; no size dependence as predicted). Caveats: regen level is a driver setting; grade is uncontrolled; because one-pedal BEVs rarely coast pedal-free, BEV windows are selected toward coast-mode cars. Highly redundant with coast_decel_low_speed_mps2 (0.87) and liftoff_decel_mps2 (0.86) but not a duplicate (different conditioning).
import numpy as np

def feat_liftoff_decel_moving(w):
    v = w["v_mps"]; a = w["a_mps2"]; gas = w["gas_pressed"]; brk = w["brake_pressed"]
    das = w["das_enabled"]; ss = w["standstill"]
    m = (np.isfinite(v) & np.isfinite(a) & (v > 3.0) & (gas < 0.5) & (brk < 0.5)
         & (das < 0.5) & (ss < 0.5))
    if np.count_nonzero(m) < 20:
        return float("nan")
    out = float(np.nanmedian(a[m]))
    return out if np.isfinite(out) else float("nan")
