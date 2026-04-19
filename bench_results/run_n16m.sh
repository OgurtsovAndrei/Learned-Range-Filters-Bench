#!/usr/bin/env bash
set -euo pipefail
cd /Users/andrei.ogurtsov/Thesis-Bench-industry

LOG="bench_results/run_n16m.log"
: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_single() {
    local test_func="$1" label="$2"
    local logfile="bench_results/${label}.log"
    log "START $label"
    if go test -v -run "$test_func" -timeout 4h -count=1 ./bench/ > "$logfile" 2>&1; then
        log "PASS  $label"
    else
        log "FAIL  $label (exit $?)"
    fi
}

log "N=16M sequential runs (sequential K, BloomARE skip, PGM skip)"
log ""

run_single "TestTradeoff_SOSD_Facebook/N=16777216" "sosd_fb_N16777216"
run_single "TestTradeoff_SOSD_OSM/N=16777216" "sosd_osm_N16777216"

log ""
log "DONE"
n_v2=$(find bench_results/data -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
log "Total v2 JSON files: $n_v2"
