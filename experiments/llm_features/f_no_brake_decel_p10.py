# Feature no_brake_decel_p10 (lens: braking_regen_coasting) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: 10th percentile of longitudinal acceleration over human-driven moving samples with the brake pedal not pressed: the strongest deceleration the drivetrain alone (regen, or engine braking plus drag) routinely produces in the window. Regenerative capacity of a BEV is about 0.1-0.25 g (and is what one-pedal drivers use for most stops), whereas closed-throttle engine braking of a combustion car in normal gears is only about 0.03-0.05 g.
# Hypothesis: BEV < ICE (more negative); HEV/PHEV between.
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, standstill, das_enabled
# Training-sample statistics: finite 0.53, AUROC BEV-vs-ICE 0.246, Spearman(footprint) -0.089
# Audit: keep_with_caveat - Routine drivetrain-only retardation; medians BEV -0.66, ICE -0.29, HEV -0.29 m/s^2 are physically coherent (mild lift-off regen in power-split hybrids; AUROC 0.25 = 0.75 in the hypothesised direction, robust to removing the dominant BEV model). Caveats: encodes the driver-selected regen level; includes grade and speed-dependent drag (Spearman +0.24 with mean speed within BEV); p10 depends on how many highway lift-offs the window contains; Spearman -0.81 with gas_pressed_decel_frac, so it adds little beyond the pedal shares.
import numpy as np

def feat_no_brake_decel_p10(w):
    # strongest (10th percentile) deceleration reached without the brake pedal while moving, human-driven
    v = w['v_mps']; a = w['a_mps2']
    g = w['gas_pressed']; b = w['brake_pressed']
    ok = np.isfinite(v) & np.isfinite(a) & np.isfinite(g) & np.isfinite(b)
    human = w['das_enabled'] < 0.5
    moving = (v > 2.0) & ~(w['standstill'] > 0.5)
    m = ok & human & moving & (b < 0.5)
    if int(m.sum()) < 50:
        return float('nan')
    x = float(np.percentile(a[m], 10))
    return x if np.isfinite(x) else float('nan')
