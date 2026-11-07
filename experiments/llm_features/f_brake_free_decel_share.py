# Feature brake_free_decel_share (lens: braking_regen_coasting) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Share of total meaningful deceleration (sum of -a over human-driven moving samples with a < -0.3 m/s^2) that is delivered with the friction-brake pedal NOT pressed. An electric drivetrain supplies 1-2 m/s^2 of regenerative retardation through the accelerator pedal or on lift-off, so most everyday slowing never involves the brake pedal; a combustion car coasts at only ~0.2 m/s^2 without the brake, so nearly all real deceleration requires it. Weighted by deceleration magnitude rather than sample count so brief hard stops count appropriately.
# Hypothesis: BEV >> ICE (BEV near 1.0, ICE near 0.15); HEV/PHEV intermediate. No size expectation.
# Channels: v_mps, a_mps2, gas_pressed, brake_pressed, standstill, das_enabled
# Training-sample statistics: finite 0.48, AUROC BEV-vs-ICE 0.827, Spearman(footprint) 0.047
# Audit: keep_with_caveat - Physically confirmed: in every ICE pseudo-model hard human decelerations (< -1.5 m/s^2) carry the brake flag (unbraked in 0-5% of samples) while BEV models are 24-95% unbraked; AUROC 0.83 holds at 0.79 without the dominant BEV model and each BEV model separates individually from ICE (0.62-0.91); HEV/PHEV vs ICE 0.59 as expected for blended regen. Caveats: (i) the -0.3 m/s^2 threshold sits at ICE closed-throttle coast-down level at highway speed, so pure-coasting ICE windows reach share 1.0 (92/564 ICE windows > 0.9; 76 of them have no sample below -1 m/s^2, median speed 23.5 m/s) - that is traffic context, not powertrain; raising the threshold to -0.6 m/s^2 removes the tail (ICE q90 1.0 -> 0.37) and lifts AUROC to 0.91 (0.88 without the dominant model) at coverage 0.40; (ii) it measures the driver-selected regen level (BEV model medians 0.28-1.0); (iii) two ICE pseudo-models (footprints 4.215 and 5.334 m^2, 45 windows) show 19-28% of hard decelerations without the brake flag, physically implausible, i.e. an unreliable per-car brake flag - a small artefact contamination (~4% of ICE windows) worth noting in the shortcut audit.
import numpy as np

def feat_brake_free_decel_share(w):
    # share of total deceleration (sum of -a over samples with a < -0.3) delivered with the brake pedal NOT pressed
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
    free = float(np.sum(-a[dec & (b < 0.5)]))
    x = free / tot
    return float(min(max(x, 0.0), 1.0)) if np.isfinite(x) else float('nan')
