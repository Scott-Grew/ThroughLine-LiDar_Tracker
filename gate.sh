#!/bin/zsh
set -e
cd "$(dirname "$0")"
[ -d build ] || cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j > /dev/null
./build/tracker_tests
