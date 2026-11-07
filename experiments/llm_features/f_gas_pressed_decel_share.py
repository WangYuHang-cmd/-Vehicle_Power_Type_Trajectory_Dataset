# Feature gas_pressed_decel_share (lens: braking_regen_coasting) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: One-pedal-driving signature: share of total deceleration (a < -0.3 m/s^2, human, moving) delivered while the accelerator is still pressed and the brake is not. In a BEV with strong regen the driver modulates deceleration by partially lifting the accelerator, so the car decelerates with gas_pressed true. In a combustion car an open throttle produces positive tractive torque; deceleration with the gas pressed essentially only occurs on steep grades or at very high speed in top gear. Distinct from brake_free_decel_share because it separates pedal-modulated regen from lift-off coasting.
# Hypothesis: BEV >> ICE (BEV ~0.7, ICE ~0.02); HEV/PHEV near ICE. No size expectation.
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, standstill, das_enabled
# Training-sample statistics: finite 0.48, AUROC BEV-vs-ICE 0.827, Spearman(footprint) 0.231
# Audit: keep_with_caveat - Magnitude-weighted one-pedal signature; ICE model medians 0.00-0.25 (mostly < 0.05) vs BEV 0.07-1.0; AUROC 0.76 without the dominant one-pedal BEV model. Same caveats as gas_pressed_decel_frac: driver-selected regen mode, per-car gas_pressed flag definition, Spearman 0.84 with the frac version (keep one or accept redundancy), and a +0.23 footprint correlation that is purely composition (the 5.58 m^2 BEV model sits at 1.0 in 183 windows) - it would act as a powertrain-to-size shortcut in a size classifier.
import numpy as np

def feat_gas_pressed_decel_share(w):
    # one-pedal signature: share of total deceleration (a < -0.3) delivered while the accelerator is pressed and the brake is not
    v = w['v_mps']; a = w['a_mps2']
    g = w['gas_pressed']; b = w['brake_pressed']
    ok = np.isfinite(v) & np.isfinite(a) & np.isfinite(g) & np.isfinite(b)
    human = w['das_enabled'] < 0.5
    moving = (v > 2.0) & ~(w['standstill'] > 0.5)
    dec = ok & human & moving & (a < -0.3)
    if int(dec.sum()) < 20:
        return float('nan')
    tot = float(np.sum(-a[dec]))
    if not np.isfinite(tot) or tot < 1e-6:
        return float('nan')
    onep = float(np.sum(-a[dec & (g > 0.5) & (b < 0.5)]))
    x = onep / tot
    return float(min(max(x, 0.0), 1.0)) if np.isfinite(x) else float('nan')
