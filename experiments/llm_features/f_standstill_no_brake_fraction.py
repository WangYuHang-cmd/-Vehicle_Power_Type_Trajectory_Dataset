# Feature standstill_no_brake_fraction (lens: energy_speed_profile) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Hold behaviour at standstill: fraction of settled human-driven standstill time (stationary for at least 1 s, no accelerator) during which the brake pedal is released. A BEV in hold mode keeps itself stationary on motor torque/auto-applied brake after a regen stop, so the driver lifts off the brake; an ICE automatic with a torque converter must be held on the brake (or it creeps). Auto-hold functions on some ICE cars weaken but do not remove the contrast.
# Hypothesis: BEV > ICE (BEV close to 1, ICE close to 0). Weak positive size dependence only through the BEV mix.
# Channels: standstill, brake_pressed, gas_pressed, das_enabled
# Training-sample statistics: finite 0.17, AUROC BEV-vs-ICE 0.735, Spearman(footprint) 0.180
# Audit: keep_with_caveat - Physically grounded (a hold-mode BEV holds itself after a regen stop; a torque-converter ICE must be held on the brake), confirmed (BEV 1.00 vs ICE 0.04, AUROC 0.73). Caveats: hold mode is a selectable setting; ICE auto-hold (a trim feature) gives the same signature; and ports that derive brake_pressed from brake pressure keep the flag at 1 under auto-hold while pedal-switch ports show 0, so part of the contrast can be port definition. Size medians reflect the BEV mix.
import numpy as np

def feat_standstill_no_brake_fraction(w):
    ss = w["standstill"]; brk = w["brake_pressed"]; gas = w["gas_pressed"]; das = w["das_enabled"]
    n = ss.shape[0]
    base = (ss > 0.5) & (das < 0.5) & (gas < 0.5)
    settled = np.zeros(n, dtype=bool)
    for i in range(10, n):
        if base[i] and np.all(ss[i - 10:i] > 0.5):    # stationary for >= 1 s already
            settled[i] = True
    if np.count_nonzero(settled) < 30:
        return float("nan")
    return float(np.mean(brk[settled] < 0.5))
