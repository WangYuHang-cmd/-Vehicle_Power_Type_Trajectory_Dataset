#!/usr/bin/env bash
# One-click, resumable download of the external datasets for the VTT transfer test (owner-run).
#
#   bash download_owner_sets.sh              # core set (about 200 GB): commaCarSegments budget selection (~78 GB), commaSteeringControl,
#                                            #   MoreTorque (Omoda E5 + CX-50 + Sonata 2020 + Mazda3 2019), comma2k19 (2 chunks)
#   bash download_owner_sets.sh --all        # core + every MoreTorque folder (347 GB) + all comma2k19 chunks (95 GB)
#   bash download_owner_sets.sh --ci         # also the openpilot CI test routes (53 GB; no stated licence: internal use only)
#   bash download_owner_sets.sh --candid     # also CANdid logs (4 GB, figshare)
#   bash download_owner_sets.sh --dry-run    # print the plan, download nothing
#   bash download_owner_sets.sh --only ccs,csc   # restrict to some steps: ccs csc moretorque comma2k19 ci candid
#
# Everything lands under $EXT/<dataset>/ (default <data-root>/external from the repository's env.sh; override with EXT=...). Re-running resumes:
# finished files are skipped, partial files continue. Logs: $EXT/logs/. Status summary: $EXT/DOWNLOAD_STATUS.md.
# Downloads are untrusted data: nothing in these folders is executed.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${VTT_REPO:-$HERE/../..}                   # this file lives in <repo>/experiments/transfer/; set VTT_REPO when running a copy elsewhere
if [ ! -f "$REPO/env.sh" ]; then echo "env.sh not found under $REPO: set VTT_REPO to the repository root" >&2; exit 2; fi
# shellcheck disable=SC1091
source "$REPO/env.sh"                           # defines D (data root), CPY (python with torch / pandas), HF_HOME under the data root
EXT=${EXT:-$D/external}
PAR=${PAR:-8}                                   # parallel curl streams for commaCarSegments
MODE=core; DRY=0; WITH_CI=0; WITH_CANDID=0; ONLY=""
for a in "$@"; do
  case "$a" in
    --all) MODE=all ;; --dry-run) DRY=1 ;; --ci) WITH_CI=1 ;; --candid) WITH_CANDID=1 ;;
    --only) shift_only=1 ;; --only=*) ONLY="${a#--only=}" ;;
    *) if [ "${shift_only:-0}" = 1 ]; then ONLY="$a"; shift_only=0; else echo "unknown option $a"; exit 2; fi ;;
  esac
done
want() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; }

# environment: HF cache on the big disk, miniconda tools (hf, huggingface-cli, gdown), $CPY interpreter
if [ -f "$REPO/env.sh" ]; then source "$REPO/env.sh"; fi
export PATH="$HOME/miniconda3/bin:$PATH"
CPY=${CPY:-$HOME/miniconda3/bin/python3}
mkdir -p "$EXT/logs"
LOG="$EXT/logs/download_$(date +%Y%m%dT%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mask() { sed -E 's/(Bearer[ =]+)[A-Za-z0-9_.-]+/\1***/g; s/hf_[A-Za-z0-9]{10,}/hf_***/g'; }   # never let an access token reach a terminal or a log
run() { if [ "$DRY" = 1 ]; then echo "DRY: $*" | mask | tee -a "$LOG"; else "$@" 2>&1 | mask | tee -a "$LOG"; return "${PIPESTATUS[0]}"; fi; }

# Hugging Face token (optional; lifts anonymous rate limits on public repos). Never printed.
TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
HFTOK=""; [ -s "$TOKEN_FILE" ] && HFTOK=$(tr -d '[:space:]' < "$TOKEN_FILE")
[ -n "$HFTOK" ] && export HF_TOKEN="$HFTOK"

df -h "$EXT" | tail -1 | awk '{print "free space on target:", $4}' | tee -a "$LOG"

# ------------------------------------------------------------------ 1. commaCarSegments (selection, ~32 GB)
step_ccs() {
  say "== commaCarSegments: index + selected segments"
  local base=https://huggingface.co/datasets/commaai/commaCarSegments/resolve/main
  mkdir -p "$EXT/ccs/segments"
  [ -s "$EXT/ccs/database.json" ] || run curl -sfL --retry 5 --retry-delay 5 -o "$EXT/ccs/database.json" "$base/database.json"
  if [ ! -s "$EXT/ccs/select.txt" ]; then
    if [ -f "$REPO/experiments/transfer/ccs_select.py" ]; then
      say "building the selection list with experiments/transfer/ccs_select.py"
      run nice -n 10 "$CPY" "$REPO/experiments/transfer/ccs_select.py" --index "$EXT/ccs/database.json" --out "$EXT/ccs/select.txt"
    else
      say "!! selection list $EXT/ccs/select.txt is missing and the selection script is not ready yet; skipping commaCarSegments (re-run later)"
      return 0
    fi
  fi
  # default list = the budget variant (held-out platforms capped at 1,000 segments, ~78 GB); CCS_FULL=1 takes the full selection (~96 GB)
  local LIST="$EXT/ccs/select_budget.txt"; { [ -s "$LIST" ] && [ "${CCS_FULL:-0}" = 0 ]; } || LIST="$EXT/ccs/select.txt"
  local n; n=$(grep -c . "$LIST"); say "selection list: $(basename "$LIST") with $n segments (about $((n * 16 / 10000)) GB)"
  [ "$DRY" = 1 ] && return 0
  export EXT base HFTOK
  # one curl per segment, resumable, skip finished files; 5 retries; errors collected, never fatal
  xargs -P "$PAR" -I{} bash -c '
    seg="{}"; out="$EXT/ccs/segments/$seg/rlog.zst"
    [ -s "$out" ] && exit 0
    mkdir -p "$(dirname "$out")"
    if [ -n "$HFTOK" ]; then auth=(-H "Authorization: Bearer $HFTOK"); else auth=(); fi
    curl -sfL -C - --retry 5 --retry-delay 5 "${auth[@]}" -o "$out" "$base/segments/$seg/rlog.zst" || { rm -f "$out"; echo "$seg" >> "$EXT/logs/ccs_failed.txt"; }
  ' < "$LIST"
  local have; have=$(find "$EXT/ccs/segments" -name rlog.zst -size +0 | wc -l)
  say "commaCarSegments: $have of $n segments on disk ($(du -sh "$EXT/ccs/segments" | cut -f1)); failures listed in logs/ccs_failed.txt (re-run to retry)"
}

# ------------------------------------------------------------------ 2. commaSteeringControl (BEV archives + ICE archives by size class, ~25 GB)
step_csc() {
  say "== commaSteeringControl: BEV archives + selected ICE archives"
  mkdir -p "$EXT/csc"
  # pick archive names from the live file list so that renamed platforms are not missed
  local list; list=$("$CPY" - <<'PY'
import re, sys
from huggingface_hub import HfApi
files = [f for f in HfApi().list_repo_files("commaai/commaSteeringControl", repo_type="dataset") if f.startswith("data/") and f.endswith(".zip")]
bev = re.compile(r"EV6|BOLT|IONIQ_5|IONIQ_6|NIRO_EV|LEAF|KONA_EL|GV60|IONIQ_EL|MACH_E|LIGHTNING|TESLA|RIVIAN|POLESTAR|ID\.?4|EQ|ETRON|MODEL_", re.I)
ice = re.compile(r"CIVIC|COROLLA|SONATA|ACCORD|CAMRY|RAM_1500|ODYSSEY|SILVERADO|SIENNA|PALISADE|HIGHLANDER|F_150|PILOT|RAV4|ELANTRA|TUCSON|SANTA_FE|EXPLORER|TELLURIDE|CRV|CR_V", re.I)
sel = [f for f in files if bev.search(f) or ice.search(f)]
sel = [f for f in sel if "SYNTHETIC" not in f.upper() and "HYBRID" not in f.upper() and "PLUG" not in f.upper()]   # hybrids are outside BEV vs ICE; keep the archive count near 30 (~25 GB)
print("\n".join(sel))
PY
)
  local n; n=$(echo "$list" | grep -c .); say "archives selected: $n"; echo "$list" | sed 's/^/   /' | tee -a "$LOG" >/dev/null
  if [ "$MODE" = all ]; then
    run hf download commaai/commaSteeringControl --repo-type dataset --local-dir "$EXT/csc" --include 'data/*.zip'
  else
    # shellcheck disable=SC2086
    run hf download commaai/commaSteeringControl --repo-type dataset --local-dir "$EXT/csc" --include $list
  fi
  say "commaSteeringControl on disk: $(du -sh "$EXT/csc" 2>/dev/null | cut -f1)"
}

# ------------------------------------------------------------------ 3. MoreTorque/rlogs (community openpilot logs; staged)
step_moretorque() {
  say "== MoreTorque/rlogs"
  mkdir -p "$EXT/moretorque"
  if [ "$MODE" = all ]; then
    run hf download MoreTorque/rlogs --repo-type dataset --local-dir "$EXT/moretorque"
  else
    run hf download MoreTorque/rlogs --repo-type dataset --local-dir "$EXT/moretorque" --include "OMODA_E5/*" "MAZDA CX-50/*" "HYUNDAI SONATA 2020/*" "MAZDA 3 2019/*"
  fi
  say "MoreTorque on disk: $(du -sh "$EXT/moretorque" 2>/dev/null | cut -f1)"
}

# ------------------------------------------------------------------ 4. comma2k19 (RAV4 + Civic chunks, 18 GB; --all = 95 GB)
step_comma2k19() {
  say "== comma2k19"
  mkdir -p "$EXT/comma2k19"
  # the hf client fetches one 9 GB zip over a single connection and can crawl; aria2c (16 connections, resumable) is used when present
  local chunks; if [ "$MODE" = all ]; then chunks=$(seq -f "Chunk_%g.zip" 1 10); else chunks="Chunk_1.zip Chunk_3.zip"; fi
  if command -v aria2c >/dev/null 2>&1; then
    mkdir -p "$EXT/comma2k19/raw_data"
    for c in $chunks; do
      [ -s "$EXT/comma2k19/raw_data/$c" ] && [ ! -f "$EXT/comma2k19/raw_data/$c.aria2" ] && continue
      run aria2c -c -x 16 -s 16 -k 8M --file-allocation=none --console-log-level=warn --summary-interval=60 \
        ${HFTOK:+--header="Authorization: Bearer $HFTOK"} -d "$EXT/comma2k19/raw_data" -o "$c" \
        "https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/$c"
    done
  elif [ "$MODE" = all ]; then
    run hf download commaai/comma2k19 --repo-type dataset --local-dir "$EXT/comma2k19" --include "raw_data/*"
  else
    run hf download commaai/comma2k19 --repo-type dataset --local-dir "$EXT/comma2k19" --include "raw_data/Chunk_1.zip" "raw_data/Chunk_3.zip"
  fi
  say "comma2k19 on disk: $(du -sh "$EXT/comma2k19" 2>/dev/null | cut -f1)"
}

# ------------------------------------------------------------------ 5. openpilot / opendbc CI test routes (optional; licence not stated: internal evaluation only)
step_ci() {
  say "== openpilot CI test routes (internal use only)"
  mkdir -p "$EXT/ci"
  run bash -c "curl -sL https://raw.githubusercontent.com/commaai/opendbc/master/opendbc/car/tests/routes.py | grep -oE 'CarTestRoute\(\"[^\"]+\",\s*[A-Z_]+\.[A-Z0-9_]+' | sed -E 's/CarTestRoute\(\"([^\"]+)\",\s*/\1\t/' > '$EXT/ci/ci_routes.tsv'"
  say "routes listed: $(grep -c . "$EXT/ci/ci_routes.tsv" 2>/dev/null || echo 0)"
  [ "$DRY" = 1 ] && return 0
  local B=https://commadataci.blob.core.windows.net/openpilotci
  while IFS=$'\t' read -r R P; do
    [ -z "$R" ] && continue
    s=0; miss=0
    while [ $miss -lt 8 ]; do
      got=0
      for f in rlog.zst rlog.bz2; do
        out="$EXT/ci/$P/${R//\//__}/$s/$f"
        if [ -s "$out" ]; then got=1; break; fi
        curl -sfL -C - --retry 3 --create-dirs -o "$out" "$B/$R/$s/$f" && { got=1; break; } || rm -f "$out"
      done
      if [ $got = 1 ]; then miss=0; else miss=$((miss + 1)); fi
      s=$((s + 1))
    done
  done < "$EXT/ci/ci_routes.tsv"
  say "CI routes on disk: $(du -sh "$EXT/ci" | cut -f1)"
}

# ------------------------------------------------------------------ 6. CANdid (optional, figshare; logs without video, ~4 GB)
step_candid() {
  say "== CANdid (figshare 29068553)"
  mkdir -p "$EXT/candid"
  run bash -c "curl -s https://api.figshare.com/v2/articles/29068553 | $CPY -c 'import json,sys; [print(f[\"download_url\"], f[\"name\"]) for f in json.load(sys.stdin)[\"files\"] if not f[\"name\"].endswith(\".mp4\")]' > '$EXT/candid/files.txt'"
  [ "$DRY" = 1 ] && return 0
  while read -r url name; do
    [ -s "$EXT/candid/$name" ] && continue
    for i in 1 2 3; do
      curl -sSL -o "$EXT/candid/$name" "$url" && ! head -c 5 "$EXT/candid/$name" | grep -q '<?xml' && break
      sleep 2
    done
  done < "$EXT/candid/files.txt"
  say "CANdid on disk: $(du -sh "$EXT/candid" | cut -f1)"
}

# ------------------------------------------------------------------ run
want ccs       && step_ccs
want csc       && step_csc
want moretorque && step_moretorque
want comma2k19 && step_comma2k19
if [ "$WITH_CI" = 1 ]     && want ci;     then step_ci; fi
if [ "$WITH_CANDID" = 1 ] && want candid; then step_candid; fi

{
  echo "# External download status ($(date +%F) $(date +%H:%M))"
  echo
  for d in ccs csc moretorque comma2k19 ci candid ultraav openacc a2d2 tesla_yuan ved microsim_ev uga_ev; do
    [ -d "$EXT/$d" ] && printf -- "- %-12s %8s  %s files\n" "$d" "$(du -sh "$EXT/$d" | cut -f1)" "$(find "$EXT/$d" -type f | wc -l)"
  done
  [ -s "$EXT/logs/ccs_failed.txt" ] && echo "- commaCarSegments failures to retry: $(sort -u "$EXT/logs/ccs_failed.txt" | wc -l) (re-run the script)"
} > "$EXT/DOWNLOAD_STATUS.md"
say "done; summary in $EXT/DOWNLOAD_STATUS.md, log $LOG"
