#!/usr/bin/env bash
# S2: run the ADAS-TO-CF extractor UNCHANGED over every chunk of routes, after reusing the earlier extraction by hardlink.
#
#   scripts/s02_extract_all.sh [--data-root DIR] [--raw-root DIR] [--dry-run] [--status] [--skip-reuse] [--only-chunk NAME] [--force]
#
# Steps: (1) reuse: hardlink probe (one route_meta.json linked inside work/signals itself, link count must be 2), then -- only
# while work/signals holds no route yet -- `cp -al -n` of config.REUSE_EXISTING_SIGNALS into work/signals (`cp -a -n` when the
# probe fails); skipped when the private salt differs from the seed salt (route_uids would not match). `$PY -m vtt.reuse --fix
# --fill` then verifies every reused dir file-for-file (names + sizes), removes incomplete ones and rebuilds every source route
# still missing (link, else copy; atomic per dir). The marker work/.reuse_signals_done is written only when every source route
# is complete in work/signals; otherwise the next run retries the reuse and this run's extractor rebuilds the rest from raw.
# (2) for every build/chunks/routes_chunk_NN.csv the extractor command of vtt.tocf_bridge.extractor_cmd (config.EXTRACT_WORKERS
# workers, --no-frame-count) with routes_chunk_long.csv last at config.EXTRACT_WORKERS_LONG workers, logs appended to
# build/logs/s02_<chunk>.log; (3) scripts/s02c_segments_table.py and scripts/s02d_accept.py (exit code = s02d's, or 2 when
# s02c itself failed). Pre-flight (real mode): private uid_map + salt + chunk dir present and build/logs/s01_summary.json not
# carrying accepted=false (exit 1; --force overrides, a missing summary is only reported).
# Resumable: re-running is safe (the extractor skips routes whose route_meta.json is current; the reuse step keeps a marker).
# --dry-run prints every command without running anything; --status is read-only (reuse state, then the extractor's own
# --dry-run per chunk: counts of routes to extract / already done); neither links, copies, removes or marks anything.
# Env overrides (tests): VTT_REUSE_EXISTING_SIGNALS, VTT_SEED_PRIVATE_ROOT (read by vtt.config). Prints counts only.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$HERE/env.sh"

DATA_ROOT="$D"; DRY=0; STATUS=0; SKIP_REUSE=0; FORCE=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --raw-root) export VTT_RAW_ROOT="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --status) STATUS=1; shift ;;
    --skip-reuse) SKIP_REUSE=1; shift ;;
    --only-chunk) ONLY="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done
export VTT_DATA_ROOT="$DATA_ROOT"
export D="$DATA_ROOT"

# every path and constant comes from vtt.config / vtt.paths (single source of truth)
read -r REUSE_SRC SEED_PRIV EXTRACTOR W_MAIN W_LONG <<<"$("$PY" -c 'from vtt import config as C; print(C.REUSE_EXISTING_SIGNALS, C.SEED_PRIVATE_ROOT, C.EXTRACTOR_SCRIPT, C.EXTRACT_WORKERS, C.EXTRACT_WORKERS_LONG)')"
read -r WORK PRIV SIG CHUNKS LOGS TMP UID_MAP <<<"$("$PY" -c 'from vtt.paths import P; print(P.WORK, P.PRIVATE, P.SIGNALS, P.CHUNKS, P.BUILD_LOGS, P.TMP, P.UID_MAP)')"

log() { echo "[s02] $*"; }
show() { echo "+ $*"; }

# ----------------------------------------------------------------------------- 0. pre-flight
s1_gate() {   # refuse to extract on top of an S1 run that did not pass its own acceptance (docs/PIPELINE.md, S1 contract note)
  local f="$LOGS/s01_summary.json"
  if [ ! -f "$f" ]; then log "S1 summary missing ($(basename "$f")): acceptance not checked"; return 0; fi
  if "$PY" -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("accepted") is True else 1)' "$f"; then
    log "S1 summary: accepted"; return 0
  fi
  if [ "$FORCE" = 1 ]; then log "S1 summary carries accepted=false: continuing (--force)"; return 0; fi
  log "S1 summary carries accepted=false: fix S1 first (or pass --force)"; exit 1
}

if [ "$DRY" = 0 ]; then
  [ -f "$UID_MAP" ] || { log "private uid_map missing ($(basename "$UID_MAP")): run S1 first"; exit 1; }
  [ -f "$PRIV/salt.txt" ] || { log "private salt missing: run S1 first"; exit 1; }
  [ -d "$CHUNKS" ] || { log "chunk dir missing: run S1 first"; exit 1; }
fi
if [ "$DRY" = 0 ] && [ "$STATUS" = 0 ]; then
  s1_gate
  mkdir -p "$SIG" "$LOGS" "$TMP"
fi

# ----------------------------------------------------------------------------- 1. reuse the earlier extraction
reuse_signals() {
  local marker="$WORK/.reuse_signals_done" mode="copy" n_src n_dst probe_src probe_dst cmd res
  if [ "$SKIP_REUSE" = 1 ]; then log "reuse: skipped (--skip-reuse)"; return 0; fi
  if [ -f "$marker" ]; then log "reuse: already done ($(cat "$marker"))"; return 0; fi
  if [ ! -d "$REUSE_SRC" ]; then log "reuse: source dir missing -> nothing reused"; return 0; fi
  if [ "$DRY" = 0 ] && ! "$PY" -c 'import sys; a, b = (open(p).read().strip() for p in sys.argv[1:3]); sys.exit(0 if a and a == b else 1)' "$PRIV/salt.txt" "$SEED_PRIV/salt.txt" 2>/dev/null; then
    log "reuse: private salt differs from the seed salt -> route_uids would not match -> nothing reused"; return 0
  fi
  n_src=$(find "$REUSE_SRC" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l)
  if [ "$DRY" = 1 ]; then
    log "reuse: would probe one hardlink of a route_meta.json inside work/signals, then link (or copy) $n_src route dirs into work/signals"
    show "cp -al -n <REUSE_EXISTING_SIGNALS>/. <work/signals>/   # or cp -a -n when the hardlink probe fails; only while work/signals holds no route"
    show "$PY -m vtt.reuse --src <REUSE_EXISTING_SIGNALS> --dst <work/signals> --fix --fill"
    return 0
  fi
  if [ "$STATUS" = 1 ]; then      # read-only: report, never link, copy, remove or mark
    log "reuse: pending ($n_src source dirs would be linked into work/signals by the next real run)"
    if [ -d "$SIG" ]; then "$PY" -m vtt.reuse --src "$REUSE_SRC" --dst "$SIG" || true; fi
    return 0
  fi
  mkdir -p "$SIG"
  n_dst=$(find "$SIG" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l)
  probe_src=$(find "$REUSE_SRC" -mindepth 2 -maxdepth 2 -name route_meta.json -print -quit)
  probe_dst="$SIG/.hardlink_probe_$$"
  rm -f "$probe_dst"
  if [ -n "$probe_src" ] && ln "$probe_src" "$probe_dst" 2>/dev/null && [ "$(stat -c %h "$probe_dst")" = "2" ]; then mode="hardlink"; fi
  rm -f "$probe_dst"
  log "reuse: $n_src route dirs in source, $n_dst already in work/signals, mode=$mode"
  if [ "$n_dst" = 0 ]; then
    if [ "$mode" = "hardlink" ]; then cmd="nice -n 15 ionice -c3 cp -al -n \"$REUSE_SRC/.\" \"$SIG/\""; else cmd="nice -n 15 ionice -c3 cp -a -n \"$REUSE_SRC/.\" \"$SIG/\""; fi
    show "$cmd"
    if ! eval "$cmd"; then log "reuse: WARNING cp returned non-zero; vtt.reuse rebuilds what is missing one directory at a time"; fi
  else
    log "reuse: work/signals is not empty -> bulk cp skipped; vtt.reuse fills the missing routes one directory at a time"
  fi
  show "$PY -m vtt.reuse --src <REUSE_EXISTING_SIGNALS> --dst <work/signals> --fix --fill"
  if res=$(nice -n 15 ionice -c3 "$PY" -m vtt.reuse --src "$REUSE_SRC" --dst "$SIG" --fix --fill); then
    echo "$res"
    echo "mode=$mode $res date=$(date -Is)" > "$marker"
    log "reuse: every source route is complete in work/signals -> marker written"
  else
    echo "$res"
    log "reuse: WARNING not every source route is complete in work/signals -> marker NOT written (the next run retries the reuse; this run's extractor rebuilds the missing routes from the raw library)"
  fi
}

# ----------------------------------------------------------------------------- 2. the extractor, chunk by chunk
extractor_cmd() {   # <routes.csv> <workers> <extractor-dry-run 0|1>
  "$PY" - "$1" "$WORK" "$PRIV" "$2" "$3" <<'PYEOF'
import shlex, sys
from pathlib import Path
from vtt.tocf_bridge import extractor_cmd
routes, work, priv, workers, dry = sys.argv[1:6]
print(shlex.join(extractor_cmd(Path(routes), Path(work), Path(priv), workers=int(workers), dry_run=bool(int(dry)))))
PYEOF
}

run_chunks() {
  local -a chunks=()
  local f name w cmd rc n_fail=0
  while IFS= read -r f; do [ -n "$f" ] && chunks+=("$f"); done < <(find "$CHUNKS" -maxdepth 1 -name 'routes_chunk_[0-9]*.csv' 2>/dev/null | sort)
  [ -f "$CHUNKS/routes_chunk_long.csv" ] && chunks+=("$CHUNKS/routes_chunk_long.csv")
  log "chunks: ${#chunks[@]} (long chunk last, $W_MAIN workers / $W_LONG for long)"
  for f in "${chunks[@]}"; do
    name=$(basename "$f" .csv)
    [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
    w=$W_MAIN; [[ "$name" == *_long ]] && w=$W_LONG
    cmd=$(extractor_cmd "$f" "$w" "$STATUS")
    show "$cmd"
    [ "$DRY" = 1 ] && continue
    if [ "$STATUS" = 1 ]; then eval "$cmd" || true; continue; fi
    set +e
    eval "$cmd" 2>&1 | tee -a "$LOGS/s02_$name.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "[s02] chunk $name exit $rc $(date -Is)" | tee -a "$LOGS/s02_$name.log"
    [ "$rc" -ne 0 ] && n_fail=$((n_fail + 1))
  done
  [ "$n_fail" -gt 0 ] && log "chunks with failed routes: $n_fail (details in work/logs/s02_failures.csv, judged by s02d)"
  return 0
}

# ----------------------------------------------------------------------------- 3. tables + acceptance
finish() {
  local c_cmd d_cmd rc_c rc_d
  c_cmd="nice -n 15 ionice -c3 $PY $R/scripts/s02c_segments_table.py --data-root \"$DATA_ROOT\" --workers $W_MAIN"
  d_cmd="$PY $R/scripts/s02d_accept.py --data-root \"$DATA_ROOT\""
  show "$c_cmd"; show "$d_cmd"
  [ "$DRY" = 1 ] && return 0
  set +e
  eval "$c_cmd" 2>&1 | tee -a "$LOGS/s02c_segments.log"
  rc_c=${PIPESTATUS[0]}
  eval "$d_cmd" 2>&1 | tee -a "$LOGS/s02d_accept.log"
  rc_d=${PIPESTATUS[0]}
  set -e
  if [ "$rc_c" -ne 0 ]; then log "s02c exit $rc_c (segments table missing or incomplete) -> exit 2"; return 2; fi
  return "$rc_d"
}

log "data root: $DATA_ROOT  extractor: $EXTRACTOR  dry-run=$DRY status=$STATUS"
reuse_signals
run_chunks
if [ "$STATUS" = 1 ]; then exit 0; fi
finish
