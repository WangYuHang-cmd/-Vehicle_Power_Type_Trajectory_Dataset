# Feature lead_decel_to_ego_decel_delay (lens: car_following_interaction) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Event-based response time: from the onset of a lead deceleration (lead accel falling below -0.6 m/s^2 after being quiet) to the ego's own deceleration onset (a < -0.5 m/s^2), when the ego is following closely (thw < 3.5 s) and not already braking. For a BEV the first driver action, lifting the accelerator, already yields substantial deceleration, and under automation the electric drive reverses torque immediately; a combustion car's lift-off gives only engine braking, so reaching -0.5 m/s^2 generally requires the brake pedal (foot transfer + hydraulic build-up), lengthening the delay. Heavier vehicles need more brake pressure to reach the same deceleration.
# Hypothesis: BEV < ICE; LARGE >= SMALL (weak).
# Channels: v_mps, a_mps2, lead_a_slope_mps2, thw_s, lead_valid
# Training-sample statistics: finite 0.33, AUROC BEV-vs-ICE 0.438, Spearman(footprint) 0.001
# Audit: keep_with_caveat - The torque-reversal mechanism is real and the separation is the same in human-only (AUROC 0.43, BEV 1.1 vs ICE 1.4 s) and automated (0.44, 1.1 vs 1.5 s) following, with no dependence on the automation share (rho 0.03). Caveats: it is not conditioned on das_enabled, so in ~54% of its windows the delay includes the controller plus per-car actuator-delay tuning (recommend requiring das == 0 over the event); driver reaction dominates in human windows; lead_a_slope onset inherits OEM radar filtering latency; 0.1-s grid.
import numpy as np

def feat_lead_decel_to_ego_decel_delay(w):
    """Median delay (s) from a lead deceleration onset (lead accel falls below -0.6 m/s^2 after
    staying above -0.6 and reaching > -0.2 in the previous 1 s; lead tracked, thw < 3.5 s,
    v > 3 m/s, ego not already decelerating) to the ego's own deceleration onset
    (a < -0.5 m/s^2) within the next 4 s. Events without an ego response are ignored."""
    v = w["v_mps"]; a = w["a_mps2"]; la = w["lead_a_slope_mps2"]; thw = w["thw_s"]
    lv = w["lead_valid"]
    delays = []
    t = 10
    while t < 590:
        if (lv[t] == 1 and np.isfinite(la[t]) and la[t] < -0.6 and np.isfinite(thw[t])
                and thw[t] < 3.5 and v[t] > 3.0):
            prev = la[t - 10:t]
            prev = prev[np.isfinite(prev)]
            cur = a[t - 5:t + 1]
            cur = cur[np.isfinite(cur)]
            if (prev.size >= 5 and np.max(prev) > -0.2 and np.min(prev) > -0.6
                    and cur.size > 0 and np.mean(cur) > -0.5):
                end = min(600, t + 41)
                seg = a[t:end]
                hit = np.where(np.isfinite(seg) & (seg < -0.5))[0]
                if hit.size > 0:
                    delays.append(hit[0] * 0.1)
                t += 40
                continue
        t += 1
    if len(delays) < 1:
        return float("nan")
    return float(np.median(delays))
