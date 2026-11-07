# Feature creep_departure_fraction (lens: energy_speed_profile) - proposed and verified by LLM agents, code kept verbatim.
# Rationale: Restart behaviour: fraction of human-driven departures from a real stop (>= 1 s standstill) where the car starts moving before the accelerator is pressed. Torque-converter ICE automatics (and Toyota-style hybrids) creep as soon as the brake is released; BEVs in hold/one-pedal mode stay still until the accelerator is pressed, so their departures are accelerator-initiated. Departures with automation on within the preceding 0.5 s are excluded because an ACC resume also launches the car without the pedal.
# Hypothesis: BEV < ICE (BEV near 0, ICE around 0.5). HEV/PHEV behave like ICE (creep emulation). No direct size dependence.
# Channels: standstill, gas_pressed, das_enabled
# Training-sample statistics: finite 0.18, AUROC BEV-vs-ICE 0.334, Spearman(footprint) -0.214
# Audit: keep_with_caveat - Sound (idle creep vs no idle torque), confirmed (ICE/HEV/PHEV 0.50 vs BEV 0.00, AUROC 0.33 i.e. 0.67 in the hypothesised direction). Caveats: creep is a selectable BEV mode (Tesla Creep) and hybrids emulate it; the result also depends on the driver's brake-release-to-accelerator timing and on port-specific standstill/gas-flag timing; only 8 distinct values.
import numpy as np

def feat_creep_departure_fraction(w):
    ss = w["standstill"]; gas = w["gas_pressed"]; das = w["das_enabled"]
    n = ss.shape[0]
    tot = 0; creep = 0
    for i in range(10, n):
        if ss[i - 1] > 0.5 and ss[i] < 0.5:
            if np.any(das[i - 5:i + 1] > 0.5):
                continue                                # ACC resume may have launched the car
            if not np.all(ss[i - 10:i] > 0.5):
                continue                                # require a real stop (>= 1 s)
            tot += 1
            if np.all(gas[i - 3:i + 1] < 0.5):          # moving before any accelerator input
                creep += 1
    if tot < 1:
        return float("nan")
    return float(creep / tot)
