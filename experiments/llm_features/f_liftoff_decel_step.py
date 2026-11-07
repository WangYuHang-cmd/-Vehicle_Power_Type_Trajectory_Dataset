# Feature liftoff_decel_step (lens: braking_regen_coasting) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Transient response to an accelerator release that is not followed by braking: mean a in the 0.6-1.5 s after the gas 1->0 transition minus mean a in the 0.5 s before it (v >= 3 m/s at release, human-driven, pedal stays released for 1.5 s, no brake and no standstill in [-0.5, +1.5] s). The step equals the retarding torque per unit mass that switches in when the pedal is released: regen torque in a BEV (large, electronically ramped in within ~0.5 s) versus closed-throttle engine braking in a combustion car (small, further softened by torque-converter slip or coasting/decoupling logic).
# Hypothesis: BEV < ICE (larger negative step).
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, standstill, das_enabled
# Training-sample statistics: finite 0.23, AUROC BEV-vs-ICE 0.387, Spearman(footprint) 0.039
# Audit: keep_with_caveat - Pre/post differencing cancels grade and drag, which is good, but the step equals (retard torque after release) minus (tractive acceleration being used before release), so it mixes the driver's throttle level into the regen/engine-braking quantity it claims to measure; regen level is also a setting. Modest separation (AUROC 0.39) consistent with physics; low correlation with liftoff_decel_mps2 (0.37) so it is not a duplicate.
import numpy as np

def feat_liftoff_decel_step(w):
    # change of longitudinal acceleration across an accelerator release: mean a 0.6-1.5 s after minus mean a 0.5 s before (no brake, human, v >= 3)
    v = w['v_mps']; a = w['a_mps2']
    g = w['gas_pressed']; b = w['brake_pressed']
    ss = w['standstill'] > 0.5
    human = w['das_enabled'] < 0.5
    okp = np.isfinite(b) & np.isfinite(g)
    gi = np.where(okp & (g > 0.5), 1, 0).astype(np.int8)
    rel = np.flatnonzero((gi[:-1] == 1) & (gi[1:] == 0)) + 1
    vals = []
    for t in rel:
        if t < 5 or t + 15 > a.shape[0]:
            continue
        seg = slice(t - 5, t + 15)
        if not okp[seg].all() or not human[seg].all() or ss[seg].any():
            continue
        if (b[seg] > 0.5).any() or (g[t:t + 15] > 0.5).any():
            continue
        if not np.isfinite(v[t]) or v[t] < 3.0:
            continue
        pre = a[t - 5:t]; post = a[t + 6:t + 15]
        pre = pre[np.isfinite(pre)]; post = post[np.isfinite(post)]
        if pre.size < 3 or post.size < 5:
            continue
        vals.append(float(np.mean(post) - np.mean(pre)))
    if len(vals) == 0:
        return float('nan')
    x = float(np.mean(vals))
    return x if np.isfinite(x) else float('nan')
