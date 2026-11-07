# Feature stop_decel_taper_ratio (lens: braking_regen_coasting) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Shape of the deceleration profile into a stop: ratio of mean a in the last 1 s before standstill to mean a 2-3 s before it (human-driven stops from > 2 m/s, with the early deceleration at least 0.3 m/s^2 so the ratio is well defined). Regenerative torque fades as motor speed falls below ~1-2 m/s and one-pedal controllers ramp the deceleration smoothly to zero at the stop, so BEV stops taper (ratio well below 1). Hydraulic friction braking in a combustion car holds its deceleration until wheel speed reaches zero (ratio nearer 1, often with a final jerk).
# Hypothesis: BEV < ICE (stronger taper in BEV).
# Channels: v_mps, a_mps2, standstill, das_enabled
# Training-sample statistics: finite 0.12, AUROC BEV-vs-ICE 0.266, Spearman(footprint) -0.121
# Audit: keep_with_caveat - Physically meaningful: the taper survives restriction to stops that used the brake pedal (AUROC 0.36, BEV 0.54 vs ICE 0.71), so it reads regen/brake blending at low speed rather than only one-pedal use. Caveats: driver brake-modulation habit (ICE drivers also feather the pedal), one-pedal controller shaping is a setting, and aEgo below ~1 m/s depends on port-specific wheel-speed dead zones.
import numpy as np

def feat_stop_decel_taper_ratio(w):
    # how much of the deceleration 2-3 s before a stop is still present in the final second (ratio of means), human-driven stops from > 2 m/s
    v = w['v_mps']; a = w['a_mps2']
    ss = w['standstill'] > 0.5
    human = w['das_enabled'] < 0.5
    si = ss.astype(np.int8)
    onsets = np.flatnonzero((si[:-1] == 0) & (si[1:] == 1)) + 1
    vals = []
    for t in onsets:
        if t < 30:
            continue
        vv = v[t - 30:t]
        if not np.isfinite(vv).all() or not (vv > 2.0).any():
            continue
        if not np.isfinite(v[t]) or v[t] >= 1.0:
            continue
        if not human[t - 30:t].all():
            continue
        early = a[t - 30:t - 20]; late = a[t - 10:t]
        early = early[np.isfinite(early)]; late = late[np.isfinite(late)]
        if early.size < 6 or late.size < 6:
            continue
        ae = float(np.mean(early)); al = float(np.mean(late))
        if ae > -0.3:
            continue
        r = al / ae
        vals.append(float(min(max(r, -2.0), 3.0)))
    if len(vals) == 0:
        return float('nan')
    x = float(np.mean(vals))
    return x if np.isfinite(x) else float('nan')
