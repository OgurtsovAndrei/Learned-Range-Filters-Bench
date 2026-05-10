#!/usr/bin/env bash
# Download all available NYC TLC trip-record parquet files.
#
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
# CDN:    https://d37ci6vzurychx.cloudfront.net/trip-data/
#
# Coverage at probe time (2026-05):
#   yellow:  2009-01 .. 2025-12   (~204 months)
#   green:   2014-12 .. 2025-12   (~133 months)
#   fhv:     2015-01 .. 2025-12   (~132 months)
#   fhvhv:   2019-12 .. 2025-12   (~73 months)
#
# Total ~542 monthly parquet files, ~25 GB.
# Idempotent: skip files already on disk. Resume-friendly.

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

BASE="https://d37ci6vzurychx.cloudfront.net/trip-data"
PARALLEL="${NYC_TAXI_PARALLEL:-2}"

download_one() {
    local fname="$1"
    if [ -f "$fname" ]; then
        return 0
    fi
    local url="$BASE/$fname"
    # CloudFront in front of nyc-tlc rate-limits aggressive bursts; keep
    # parallelism low (default 2) and retry on transient errors.
    if curl -fsSL --max-time 600 --retry 4 --retry-delay 10 \
            --retry-all-errors -o "${fname}.tmp" "$url"; then
        mv "${fname}.tmp" "$fname"
        echo "  ok   $fname"
    else
        rm -f "${fname}.tmp"
        echo "  FAIL $fname" >&2
    fi
}
export -f download_one
export BASE

generate_targets() {
    for y in $(seq 2009 2025); do
        for m in $(seq -w 1 12); do
            echo "yellow_tripdata_${y}-${m}.parquet"
        done
    done
    for y in $(seq 2014 2025); do
        for m in $(seq -w 1 12); do
            [ "$y" = "2014" ] && [ "$m" != "12" ] && continue
            echo "green_tripdata_${y}-${m}.parquet"
        done
    done
    for y in $(seq 2015 2025); do
        for m in $(seq -w 1 12); do
            echo "fhv_tripdata_${y}-${m}.parquet"
        done
    done
    for y in $(seq 2019 2025); do
        for m in $(seq -w 1 12); do
            [ "$y" = "2019" ] && [ "$m" != "12" ] && continue
            echo "fhvhv_tripdata_${y}-${m}.parquet"
        done
    done
}

echo "NYC TLC trip-data downloader"
echo "  base:        $BASE"
echo "  parallelism: $PARALLEL (override via NYC_TAXI_PARALLEL=N)"
echo

generate_targets | xargs -P "$PARALLEL" -I {} bash -c 'download_one "$@"' _ {}

echo
parquet_count=$(find . -maxdepth 1 -name '*.parquet' -type f | wc -l | tr -d ' ')
echo "Done. $parquet_count parquet files on disk."
du -sh "$DIR" 2>/dev/null || true
