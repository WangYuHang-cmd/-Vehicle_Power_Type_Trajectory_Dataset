# Feature gas_pressed_decel_frac (lens: longitudinal_powertrain) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Fraction of human gas-pressed samples at >= 3 m/s in which the car nevertheless decelerates faster than 0.3 m/s^2. A combustion drivetrain only adds torque when the pedal is pressed, so deceleration under gas needs a steep uphill; a one-pedal BEV's accelerator is bipolar (its lower travel commands regen), so marked deceleration while the pedal flag is on is routine.
# Hypothesis: BEV > ICE
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, das_enabled
# Training-sample statistics: finite 0.49, AUROC BEV-vs-ICE 0.808, Spearman(footprint) 0.165
# Audit: keep_with_caveat - The bipolar-pedal signature is real: every ICE pseudo-model sits at 0.00-0.05 while BEV models span 0.08-0.28, and AUROC stays 0.76 after removing the dominant one-pedal BEV model (365 windows). Caveats: (i) it measures the regen mode the driver selected (one-pedal on/off, regen level) as much as the drivetrain, so BEVs driven without one-pedal look ICE-like; (ii) gas_pressed is a per-car software-thresholded flag in the recording stack (mostly 'any pedal travel' but not uniform; gas-pressed-while-moving fraction ranges 0.50-0.96 across BEV models); (iii) Spearman 0.84 with gas_pressed_decel_share, near-redundant; (iv) its +0.17 footprint correlation is composition (the largest BEV model is the strongest one-pedal car), not physics - do not let it serve as a size cue; report size performance within powertrain.
import numpy as np

def feat_gas_pressed_decel_frac(w):
    """Fraction of human gas-pressed samples (v >= 3 m/s) in which the car decelerates by more than 0.3 m/s^2: a bipolar (drive/regen) accelerator pedal."""
    v = np.asarray(w["v_mps"], dtype=np.float64)
    a = np.asarray(w["a_mps2"], dtype=np.float64).copy()
    a[np.abs(a) > 8.0] = np.nan
    gi = np.nan_to_num(np.asarray(w["gas_pressed"], dtype=np.float64), nan=0.0) > 0.5
    bi = np.nan_to_num(np.asarray(w["brake_pressed"], dtype=np.float64), nan=1.0) > 0.5
    di = np.nan_to_num(np.asarray(w["das_enabled"], dtype=np.float64), nan=1.0) > 0.5
    m = gi & (~bi) & (~di) & (v >= 3.0) & np.isfinite(a)
    if m.sum() < 30:
        return float("nan")
    return float(np.mean(a[m] < -0.3))
