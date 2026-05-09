#!/bin/bash
set -e

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "=== Switching to FIXED Grafite (Optimized) ==="

cd "$DIR/ext/grafite"
echo "Checking out fixed Grafite hash (37d8bd1)..."
git checkout 37d8bd1e60647b2af3d062d691b40d20be381b77

echo "Checking out fixed SUX hash (5ff0530)..."
cd lib/sux
git checkout 5ff0530a942a50097fb7281ac9a58cca781329fe

cd "$DIR/build"
echo "Rebuilding C++ wrapper..."
make clean && make -j8

echo "----------------------------------------------------"
echo "DONE. Grafite is now in FIXED mode."
echo "Running benchmarks will now show O(log N) latency on clusters."
