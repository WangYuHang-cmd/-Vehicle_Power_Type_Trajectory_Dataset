# Feature closing_decel_nobrake_frac (lens: car_following_interaction) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Which channel sheds speed when the ego closes on a slower lead. Among human-driven samples with a < -0.4 m/s^2 while closing, the fraction without the brake pedal pressed. A drivetrain with strong regenerative braking (lift-off or one-pedal) lets the driver manage the gap without touching the friction brakes; a combustion car needs the brake pedal for anything beyond engine braking, so most of its closing decelerations show brake_pressed = 1. The gas-pressed variant (one-pedal) and the no-pedal variant (lift-off regen) are both counted as 'no brake'.
# Hypothesis: BEV > ICE (BEV close to 1, ICE close to 0). Size: no expectation.
# Channels: v_mps, a_mps2, brake_pressed, das_enabled, standstill, lead_valid, lead_v_rel_mps
# Training-sample statistics: finite 0.25, AUROC BEV-vs-ICE 0.877, Spearman(footprint) -0.010
# Audit: keep_with_caveat - Strongest BEV/ICE separator (AUROC 0.88, BEV 1.00 vs ICE 0.12) with sound physics (speed shed through the drivetrain vs friction brake). Caveats: lift-off/one-pedal regen strength is a driver setting, drivers differ in coasting vs braking habit, and brake_pressed is derived differently across ports (pedal switch vs pressure threshold). HEV/PHEV sit in between as expected (0.31/0.28).
import numpy as np

def feat_closing_decel_nobrake_frac(w):
    """Among human-driven samples where the ego is decelerating (a < -0.4 m/s^2, v > 3 m/s)
    while closing on a tracked lead (lead_v_rel < -0.3 m/s), the fraction with the brake pedal
    NOT pressed, i.e. speed shed through the drivetrain (lift-off / one-pedal regen) rather than
    the friction brakes."""
    v = w["v_mps"]; a = w["a_mps2"]
    sel = ((a < -0.4) & (v > 3.0) & (w["das_enabled"] == 0) & (w["standstill"] == 0)
           & (w["lead_valid"] == 1) & (w["lead_v_rel_mps"] < -0.3) & np.isfinite(a))
    if int(np.sum(sel)) < 20:
        return float("nan")
    return float(np.mean(w["brake_pressed"][sel] == 0))
