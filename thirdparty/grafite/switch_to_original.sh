#!/bin/bash
set -e

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "=== Switching to ORIGINAL Grafite (Pathological) ==="

cd "$DIR/ext/grafite"
echo "Checking out original Grafite hash (6838e9c)..."
git checkout 6838e9c6f757b300c410e79bc9c608ece2c5f23f

echo "Updating submodules (restoring original SUX)..."
git submodule update --init --recursive

cd "$DIR/build"
echo "Rebuilding C++ wrapper..."
make clean && make -j8

echo "----------------------------------------------------"
echo "DONE. Grafite is now in ORIGINAL mode."
echo "Running benchmarks will now show O(N) latency on clusters."
