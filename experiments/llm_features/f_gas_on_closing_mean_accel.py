# Feature gas_on_closing_mean_accel (lens: car_following_interaction) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Mean acceleration while the accelerator is pressed, human-driven, closing on a lead at > 0.5 m/s. A combustion drivetrain cannot produce negative wheel torque with the accelerator applied, so this mean is near zero or positive for ICE (mild closing while still on the gas). A BEV in one-pedal / strong-regen mode modulates deceleration with a partially pressed accelerator, so the ego decelerates while gas_pressed is 1. This is a torque-sign capability of the drivetrain, not a driver preference.
# Hypothesis: BEV < ICE (BEV negative, ICE around zero or slightly positive). Size: no expectation.
# Channels: v_mps, a_mps2, gas_pressed, das_enabled, standstill, lead_valid, lead_v_rel_mps
# Training-sample statistics: finite 0.25, AUROC BEV-vs-ICE 0.208, Spearman(footprint) -0.048
# Audit: keep_with_caveat - Sound drivetrain capability (negative wheel torque with the accelerator partly pressed), confirmed (BEV -0.16 vs ICE +0.09, AUROC 0.21). Caveats: whether a partial pedal regenerates depends on the selected regen mode, gas_pressed is a port-defined threshold flag, and the lead/closing condition is only a situation selector (fine) but limits coverage to radar/vision-lead windows.
import numpy as np

def feat_gas_on_closing_mean_accel(w):
    """Mean ego acceleration while the accelerator pedal is pressed, human-driven, v > 3 m/s,
    lead tracked and closing at > 0.5 m/s. A drivetrain that can produce negative wheel torque
    under a partially pressed accelerator (one-pedal regen) gives negative values; a combustion
    drivetrain cannot decelerate with the accelerator applied."""
    v = w["v_mps"]; a = w["a_mps2"]
    sel = ((w["gas_pressed"] == 1) & (w["das_enabled"] == 0) & (w["standstill"] == 0) & (v > 3.0)
           & (w["lead_valid"] == 1) & (w["lead_v_rel_mps"] < -0.5) & np.isfinite(a))
    if int(np.sum(sel)) < 15:
        return float("nan")
    val = float(np.mean(a[sel]))
    return val if np.isfinite(val) else float("nan")
