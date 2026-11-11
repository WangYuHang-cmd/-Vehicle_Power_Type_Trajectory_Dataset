#!/usr/bin/env bash
# Zero-shot scoring campaign over every external window set (experiments/transfer/RESULTS.md).
# Runs score.py once per (dataset, checkpoint set, mask) with the matched reduced-input variant first and the full
# 20-channel models under the same channel mask as the secondary number. Sequential, nice -n 10, GPU when free.
#   source env.sh && bash experiments/transfer/run_scoring.sh [datasets...]      (default: the phase-1 sets below; csc, moretorque, ccs, ccs_honda, comma2k19
#                                                                                  and the understeer-corrected re-decodes ccs_us, ccs_honda_us, comma2k19_us by name)
set -u
cd "$(dirname "$0")/../.." || exit 1
source env.sh
S="nice -n 10 $CPY experiments/transfer/score.py"
B=${BOOTSTRAP:-1000}
V2="transfer__bev_ice__resnet__V2__s*,transfer__size__gru__V2__s*"
V3="transfer__bev_ice__resnet__V3__s*,transfer__size__gru__V3__s*"
V1="transfer__bev_ice__resnet__V1__s*,transfer__size__gru__V1__s*"
LOG=${LOG:-$D/external/logs/scoring_$(date +%Y%m%d_%H%M).log}
run() { echo "### $(date +%H:%M:%S) $*" | tee -a "$LOG"; $S "$@" --bootstrap "$B" >>"$LOG" 2>&1 || echo "!!! FAILED: $*" | tee -a "$LOG"; }
sets=("$@"); [ ${#sets[@]} -eq 0 ] && sets=(ultraav openacc microsim_ev uga_ev tesla_yuan openacc_500x ved a2d2 ccs_sample)
for ds in "${sets[@]}"; do
  case $ds in
    ultraav|openacc|microsim_ev|uga_ev)         # longitudinal-only sets: V2 variant (primary), V3 variant (size reference), full models masked
      run --dataset $ds --ckpts "$V2"
      run --dataset $ds --ckpts "$V3"
      run --dataset $ds --mask-channels V2
      run --dataset $ds --mask-channels V3 ;;
    tesla_yuan|openacc_500x|ved)                 # speed-only sets: V3 variant (primary), full models with the V3 mask
      run --dataset $ds --ckpts "$V3"
      run --dataset $ds --mask-channels V3 ;;
    a2d2)                                        # lateral + speed + pedals, lead and torque already absent: full models (primary), V1 variant, masks
      run --dataset $ds
      run --dataset $ds --ckpts "$V1"
      run --dataset $ds --mask-channels V1
      run --dataset $ds --ckpts "$V3"
      run --dataset $ds --mask-channels V3 ;;
    ccs_sample)                                  # all 20 channels: full models only (4 windows, pipeline check)
      run --dataset $ds ;;
    csc)                                         # commaSteeringControl: the V1 channel list (lateral + speed), one window per segment, 36 archives
      run --dataset $ds --ckpts "$V1"            #   V1 variants (primary, matched inputs)
      run --dataset $ds --mask-channels V1       #   full models under the same mask (secondary)
      run --dataset $ds --ckpts "$V3" ;;         #   speed-only variants (powertrain reference; size for comparison)
    moretorque)                                  # MoreTorque rlogs: all 20 channels (vision lead, localizer yaw): full models (primary), variants and masks for comparison
      run --dataset $ds
      run --dataset $ds --ckpts "$V1"
      run --dataset $ds --ckpts "$V2"
      run --dataset $ds --ckpts "$V3"
      run --dataset $ds --mask-channels V1
      run --dataset $ds --mask-channels V2
      run --dataset $ds --mask-channels V3 ;;
    ccs|ccs_honda|comma2k19|ccs_us|ccs_honda_us|comma2k19_us)
                                                 # commaCarSegments, device-disjoint selection: all 20 channels (radar / stock-ACC / no lead by platform, steering-derived yaw);
                                                 #   ccs_honda = the nine Honda platforms re-decoded with the gearbox option of ccs_decode.py (same nine runs);
                                                 #   comma2k19 = two cars on 2018 hardware decoded from the raw 2018 CAN by the same decoder (same nine runs);
                                                 #   *_us = the same three sets re-decoded with `--yaw derived-understeer --understeer-k K` (same nine runs; collect_scores.py
                                                 #   compares each with its original in the "Understeer-corrected yaw re-decodes" section)
      run --dataset $ds                          #   full models (primary)
      run --dataset $ds --ckpts "$V1"            #   reduced-input variants for comparison
      run --dataset $ds --ckpts "$V2"
      run --dataset $ds --ckpts "$V3"
      run --dataset $ds --mask-channels V1       #   full models under the variant masks (secondary)
      run --dataset $ds --mask-channels V2
      run --dataset $ds --mask-channels V3
      run --dataset $ds --mask-channels yaw      #   sensitivity: yaw / curvature / lateral acceleration are steering-derived here (-> score__*__mask-yaw.*)
      run --dataset $ds --mask-channels lead ;;  #   sensitivity: vision-only platforms carry no lead (-> score__*__mask-lead.*)
    *) echo "unknown set $ds" | tee -a "$LOG" ;;
  esac
done
echo "### $(date +%H:%M:%S) done" | tee -a "$LOG"
