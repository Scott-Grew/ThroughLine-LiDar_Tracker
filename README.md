# tracker

A C++ real-time 3D multi-object tracker and replay viewer on the Waymo Open
Dataset (Perception, v2 parquet). A Kalman filter with constant-turn-rate
motion tracks detections across frames, a Hungarian assignment matches
detections to tracks, a perturbation stage injects dropout, position and yaw
noise, and latency into the detection stream, CLEAR tracking metrics (MOTA,
MOTP) score the result against ground truth, and Waymo's official evaluator
is the referee for any reported number.

## Build

Prerequisites (installed via Homebrew): `cmake`, `protobuf`.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This fetches Eigen 3.4.0, Catch2 v3.7.1, GLFW 3.4, Dear ImGui v1.91.5 and
ImPlot v0.16 via CMake FetchContent on first configure.

## Stage

The stager reads Waymo's v2 parquet components and writes a segment log in
the binary format below. It runs in its own Python virtual environment, kept
outside the iCloud-synced `Documents` tree:

```
python3 -m venv ~/venvs/tracker
~/venvs/tracker/bin/pip install pyarrow numpy
~/venvs/tracker/bin/python stage/stage_segment.py --parquet-root ROOT --segment SEGMENT --out OUT
```

## Run

```
./build/tracker --segment PATH [--source gt|det] [--assign hungarian|greedy]
                [--dropout PROBABILITY] [--noise METRES] [--latency MILLISECONDS]
                [--seed N] [--rate R] [--headless] [--export PATH] [--help]
```

`--headless` runs the tracker without the viewer and prints per-class MOTA
and MOTP plus the tracker step p50/p99 and overrun count. Without
`--headless` the viewer opens alongside the tracker running on its own
thread.

## Test

```
./gate.sh
```

## Log format

Segment logs are little-endian binary files.

Header: magic `TRKLOG01` (8 bytes), `uint32` frame count, `uint32` name
length, name bytes.

Per frame: `int64` capture time in microseconds; 16 `float64` values for
`vehicle_to_world`, row-major; `uint32` point count followed by that many
points, each 4 `float32` values (x, y, z, intensity); `uint32` ground truth
box count followed by, per box: `uint64` object id, `uint8` class, 7
`float64` values (center x, center y, center z, length, width, height, yaw),
`int32` lidar points in box, `uint8` tracking difficulty; `uint32` detection
count followed by, per detection: `uint8` class, 7 `float64` box values,
`float32` score.

## Results

No numbers yet.
