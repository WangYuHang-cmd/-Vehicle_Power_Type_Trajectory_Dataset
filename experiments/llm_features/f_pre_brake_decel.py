# Feature pre_brake_decel (lens: braking_regen_coasting) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Deceleration already present during the 0.6 s before the brake pedal is first applied (brake 0->1 transition at v >= 3 m/s, human-driven and not standstill throughout the lead-in). In a BEV the friction brake is added on top of regen that is already slowing the car (lift-off regen or one-pedal regen), so a at brake onset is strongly negative; in a combustion car the brake must supply nearly all the deceleration, so the pre-brake value is only the small engine-braking level, or about zero when the driver comes straight off the throttle. Averaged over all qualifying brake onsets in the window.
# Hypothesis: BEV < ICE (more negative); HEV/PHEV between.
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, standstill, das_enabled
# Training-sample statistics: finite 0.24, AUROC BEV-vs-ICE 0.172, Spearman(footprint) 0.056
# Audit: keep_with_caveat - Sound: deceleration already present before the brake is applied is the lift-off retard torque (regen vs engine braking); confirmed (BEV -0.72 vs ICE -0.23, AUROC 0.17). Caveat: the regen level at lift-off is a driver-selectable setting (standard/low/one-pedal) and the value also depends on whether the driver coasts before braking; grade is uncontrolled.
import numpy as np

def feat_pre_brake_decel(w):
    # deceleration already present during the 0.6 s before the brake pedal is first applied (v >= 3 m/s, human-driven)
    v = w['v_mps']; a = w['a_mps2']
    g = w['gas_pressed']; b = w['brake_pressed']
    ss = w['standstill'] > 0.5
    human = w['das_enabled'] < 0.5
    okp = np.isfinite(b) & np.isfinite(g)
    bi = np.where(okp & (b > 0.5), 1, 0).astype(np.int8)
    onsets = np.flatnonzero((bi[:-1] == 0) & (bi[1:] == 1)) + 1
    vals = []
    for t in onsets:
        if t < 6:
            continue
        seg = slice(t - 6, t)
        if not okp[seg].all() or not human[seg].all() or ss[seg].any():
            continue
        if not np.isfinite(v[t]) or v[t] < 3.0:
            continue
        aa = a[seg]
        aa = aa[np.isfinite(aa)]
        if aa.size < 4:
            continue
        vals.append(float(np.mean(aa)))
    if len(vals) == 0:
        return float('nan')
    x = float(np.mean(vals))
    return x if np.isfinite(x) else float('nan')
