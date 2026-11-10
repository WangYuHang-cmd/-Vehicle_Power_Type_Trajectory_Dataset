#!/usr/bin/env bash
# Fetch the small open external datasets for the VTT transfer test (CANDIDATES.md section 5).
# Usage: fetch_open_sets.sh <step> [<step> ...]   steps: openacc tesla_yuan ved microsim_ev uga_ev
# Downloads go to $EXT/<name>/raw (untrusted data: nothing in there is ever executed);
# logs go to $EXT/<name>/work/fetch/. Run one step at a time (nice -n 10, 20 MB/s cap).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${VTT_REPO:-$HERE/../..}                   # this file lives in <repo>/experiments/transfer/
if [ ! -f "$REPO/env.sh" ]; then echo "env.sh not found under $REPO: set VTT_REPO to the repository root" >&2; exit 2; fi
# shellcheck disable=SC1091
source "$REPO/env.sh"                           # defines D (data root) and CPY (python with pandas)
EXT=${VTT_EXTERNAL:-$D/external}
RATE=20m
NICE="nice -n 10"
CPY=${CPY:?}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

step_openacc() {
  local raw=$EXT/openacc/raw work=$EXT/openacc/work/fetch
  mkdir -p "$raw" "$work"
  log "openacc: wget -r of the JRC open directory (7 campaigns)"
  # path = ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/<campaign>/... -> cut 5 dirs so campaigns sit in raw/
  ( cd "$raw" && $NICE wget -r -np -nH --cut-dirs=5 -R 'index.html*' -e robots=off \
      --limit-rate=$RATE --tries=5 --waitretry=10 --no-verbose -a "$work/wget.log" \
      'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/' )
  log "openacc: wget exit $?"
  find "$raw" -name 'index.html*' -delete
}

step_tesla_yuan() {
  local raw=$EXT/tesla_yuan/raw work=$EXT/tesla_yuan/work/fetch
  mkdir -p "$raw" "$work"
  log "tesla_yuan: figshare file 50103315 (article 23904102 v2)"
  curl -s -m 60 https://api.figshare.com/v2/articles/23904102 -o "$work/article.json"
  $NICE wget --limit-rate=$RATE --tries=5 --waitretry=10 --no-verbose -a "$work/wget.log" \
      -O "$raw/DATA_S1_update20241031.zip" 'https://ndownloader.figshare.com/files/50103315'
  log "tesla_yuan: wget exit $?"
  ( cd "$raw" && md5sum DATA_S1_update20241031.zip | tee "$work/md5.txt" )
  grep -q 628a3f0347662de30d958ada5d611f3e "$work/md5.txt" && log "tesla_yuan: md5 OK" || log "tesla_yuan: md5 MISMATCH"
}

step_ved() {
  local raw=$EXT/ved/raw work=$EXT/ved/work/fetch
  mkdir -p "$raw" "$work"
  log "ved: git clone --depth 1 gsoh/VED"
  $NICE git clone --depth 1 https://github.com/gsoh/VED.git "$raw/VED" >> "$work/git.log" 2>&1
  log "ved: clone exit $?"
  ( cd "$raw/VED" && git rev-parse HEAD > "$work/commit.txt" && git log -1 --format='%H %cI %s' >> "$work/git.log" )
  for a in "$raw"/VED/Data/*.7z; do
    local stem; stem=$(basename "$a" .7z)
    mkdir -p "$raw/extracted/$stem"
    log "ved: 7z x $(basename "$a")"
    $NICE 7z x -y -o"$raw/extracted/$stem" "$a" >> "$work/7z.log" 2>&1
    log "ved: 7z exit $?"
  done
}

step_microsim_ev() {
  local raw=$EXT/microsim_ev/raw work=$EXT/microsim_ev/work/fetch
  mkdir -p "$raw" "$work"
  log "microsim_ev: git clone microSIM-ACC/EV"
  $NICE git clone https://github.com/microSIM-ACC/EV.git "$raw/EV" >> "$work/git.log" 2>&1
  log "microsim_ev: clone exit $?"
  ( cd "$raw/EV" && git rev-parse HEAD > "$work/commit.txt" && git log -1 --format='%H %cI %s' >> "$work/git.log" )
  find "$raw/EV" -maxdepth 2 -name '*.zip' -print0 | while IFS= read -r -d '' z; do
    local stem; stem=$(basename "$z" .zip)
    mkdir -p "$raw/unzipped/$stem"
    log "microsim_ev: unzip $(basename "$z")"
    $NICE unzip -q -o "$z" -d "$raw/unzipped/$stem" >> "$work/unzip.log" 2>&1
    log "microsim_ev: unzip exit $?"
  done
}

step_uga_ev() {
  local raw=$EXT/uga_ev/raw work=$EXT/uga_ev/work/fetch
  mkdir -p "$raw" "$work"
  log "uga_ev: gdown --folder (Google Drive, anyone with the link)"
  local ok=0
  for attempt in 1 2; do
    $NICE $CPY -m gdown --folder 'https://drive.google.com/drive/folders/18L-EAKB0W2g7jJbiFH8zRxq0K0Toa28y' \
        -O "$raw" >> "$work/gdown_attempt$attempt.log" 2>&1 && { ok=1; break; }
    log "uga_ev: attempt $attempt failed"
    sleep 20
  done
  [ $ok = 1 ] && log "uga_ev: gdown OK" || log "uga_ev: gdown FAILED twice"
}

for s in "$@"; do
  case "$s" in
    openacc|tesla_yuan|ved|microsim_ev|uga_ev) "step_$s" ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
