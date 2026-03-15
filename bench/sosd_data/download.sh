#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

download() {
    local name="$1"
    local url="$2"

    if [ -f "$name" ]; then
        echo "$name already exists, skipping"
        return
    fi

    echo "Downloading $name.zst ..."
    curl -L -o "${name}.zst" "$url"

    echo "Decompressing $name.zst ..."
    zstd -d "${name}.zst"
    rm "${name}.zst"

    echo "$name ready"
}

download "fb_200M_uint64" \
    "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/EATHF7"

download "wiki_ts_200M_uint64" \
    "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/SVN8PI"

download "osm_cellids_800M_uint64" \
    "https://www.dropbox.com/s/j1d4ufn4fyb4po2/osm_cellids_800M_uint64.zst?dl=1"

download "books_200M_uint32" \
    "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/5YTV8K"

echo "All datasets downloaded."
