# tracker

A C++ real-time 3D multi-object tracker and replay viewer on the Waymo Open
Dataset (Perception, v2 parquet). A Kalman filter with constant-turn-rate
motion tracks detections across frames, a Hungarian assignment matches
detections to tracks, a perturbation stage injects dropout, position noise,
and latency into the detection stream, CLEAR tracking metrics (MOTA,
MOTP) score the result against ground truth, and Waymo's official evaluator
is the referee for any reported number.

## Build

Prerequisites (installed via Homebrew): `cmake`, `protobuf`.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This fetches Eigen 3.4.0, Catch2 v3.7.1, GLFW 3.4, Dear ImGui v1.91.5 and the
Foxglove C++ SDK v0.27.0 (the Intel Mac artifact) via CMake FetchContent on
first configure. The Foxglove dist ships a prebuilt C library and the C++
wrapper as source, with a hand-written CMake package config rather than a
top-level `CMakeLists.txt`, so it is populated with `FetchContent_Populate`
and built with `find_package(foxglove-sdk)` plus the config's own
`foxglove_sdk_add_cpp_library()` helper. Its static archive leaves zstd and
lz4 as unresolved symbols for the final link, so both are a system
dependency here, installed via Homebrew (`brew install zstd lz4`) rather than
fetched - the SDK's own CMake does not fetch them. It needs no protobuf or
nlohmann_json: message encoding happens inside the prebuilt library, not in
the C++ wrapper. Only the `tracker` executable links the Foxglove SDK;
`tracker_core` and `tracker_tests` stay free of it. Nothing is installed
system-wide beyond zstd and lz4. This first build - fetching every dependency
and compiling `tracker` and `tracker_tests` from a clean `build/` directory -
is a one-off cost of 274 seconds under `-j2` on this machine.

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
./build/tracker --segment PATH [--assign hungarian|greedy]
                [--dropout PROBABILITY] [--noise METRES] [--latency MILLISECONDS]
                [--sigma-position METRES] [--sigma-yaw RADIANS]
                [--seed N] [--rate R] [--headless] [--record PATH]
                [--export PATH] [--help]
```

`--sigma-position` and `--sigma-yaw` set the filter's assumed measurement
noise (defaults 0.1 m and 0.02 rad); the stager prints its own measured
values for a staged segment, which are the numbers to pass here.

`--headless` runs the tracker without the viewer and prints per-class MOTA,
MOTP, id switches, misses and false positives, plus the tracker step
p50/p99, overrun count and frame count. Without `--headless` the viewer
opens alongside the tracker running on its own thread: a small ImGui window
of perturbation sliders and the same running metrics and timing numbers the
headless summary prints. The window draws none of the scene itself - `--record
PATH` writes a gapless `.mcap` recording of the whole run afterward, built
from the same stored tracks the `--export` CSV comes from, containing the
lidar points, the ego box, and every confirmed track's box, history trail,
predicted path and uncertainty ellipse. This works whether or not `--headless`
was given.

## Test

```
./gate.sh
```

## Viewing a recording

`--record PATH` writes an MCAP file using the Foxglove C++ SDK's well-known
schemas (`foxglove.PointCloud`, `foxglove.SceneUpdate`, and a `/tf` topic of
`foxglove.FrameTransform` placing the ego frame inside the world frame). Open
it in [Lichtblick](https://github.com/lichtblick-suite/lichtblick/releases), a
Foxglove Studio fork with a mac-universal build and no account needed: File
-> Open local file, choose the `.mcap`, add a 3D panel, and set its display
frame to `world` (or `ego` to have the camera follow the vehicle).

## Log format

Segment logs are little-endian binary files.

Header: magic `TRKLOG02` (8 bytes), `uint32` frame count, `uint32` name
length, name bytes.

Per frame: `int64` capture time in microseconds; 16 `float64` values for
`vehicle_to_world`, row-major; `uint32` point count followed by that many
points, each 3 `float32` values (x, y, z); `uint32` ground truth box count
followed by, per box: `uint64` object id, `uint8` class, 7 `float64` values
(center x, center y, center z, length, width, height, yaw), `int32` lidar
points in box.

The stager prints a sanity check for the first frame of every segment it
stages: the number of staged points that fall inside each ground truth box,
divided by the number of points Waymo itself reports for those boxes. A
ratio far from 1 (outside 0.5-1.5) means the range image's azimuth
convention is wrong, and the stager prints a warning rather than staging
the segment silently.

## Export format

`--export PATH` writes one CSV row per confirmed track per frame, in the
vehicle frame of that frame rather than the tracker's internal world frame:
`frame_timestamp_micros,track_id,object_class,center_x,center_y,center_z,length,width,height,yaw`.
`eval/to_waymo_objects.py` turns this into Waymo's Objects proto for the
official evaluator; see "Waymo's official evaluator" below.

## Results

Scored by Waymo's own `compute_tracking_metrics_main`, built from their repository and run in a
linux/amd64 container. `eval/run_official.sh` produces every number below. Nothing in this project
takes part in the scoring.

Three validation segments from Waymo Open Dataset Perception v2, 198/199/198 frames at 10 Hz.
Detections are the labelled boxes the sensor returned at least one point for, so this measures the
tracker with detection held perfect. It is not comparable to a challenge submission, which scores a
detector and a tracker together.

MOTA at LEVEL_2, which includes every labelled object:

| segment | class | dropout 0 | dropout 0.2 | dropout 0.4 |
|---|---|---|---|---|
| 10203656353524179475 | vehicle | 0.877 | 0.845 | 0.760 |
| 1024360143612057520 | vehicle | 0.956 | 0.942 | 0.873 |
| 10247954040621004675 | vehicle | 0.955 | 0.936 | 0.864 |
| 10203656353524179475 | pedestrian | 0.783 | 0.738 | 0.591 |
| 1024360143612057520 | pedestrian | 0.911 | 0.903 | 0.817 |
| 10247954040621004675 | pedestrian | 0.734 | 0.742 | 0.664 |

Segment 1024360143612057520 also contains cyclists: MOTA 0.864 at dropout 0.

By range, vehicles at dropout 0, showing where tracking actually fails:

| segment | 0 to 30 m | 30 to 50 m | beyond 50 m |
|---|---|---|---|
| 10203656353524179475 | 0.969 | 0.964 | 0.771 |
| 1024360143612057520 | 0.972 | 0.967 | 0.921 |
| 10247954040621004675 | 0.982 | 0.969 | 0.905 |

Distant objects return few points, their boxes jitter more, and the association gate rejects them.
Inside 50 m the tracker is close to the labels; beyond it the measurement itself is the limit.

Tracker step time: 0.07 ms median, worst p99 0.632 ms over 36 runs, against the 100 ms a frame
allows at 10 Hz. No frame exceeded the budget.

Every row came from:

```
./build/tracker --segment ~/waymo-data/staged/SHORT.trklog --headless \
    --sigma-position 0.036 --sigma-yaw 0.0088 --dropout P --seed 1 --export TRACKS.csv
./eval/run_official.sh FULL_SEGMENT_NAME TRACKS.csv
```

Two independent implementations agree on these tracks: Waymo's evaluator and `motmetrics` 1.4.0
match to within 0.0005 MOTA on every segment and class, and this project's own `metrics.cpp`
monitor matches both exactly, including the integer miss, false-positive and switch counts. The
monitor exists to watch a run as it happens and is never the source of a reported number.


## Waymo's official evaluator

`eval/run_official.sh` prints Waymo's own 3D tracking metrics for one segment, produced by
`compute_tracking_metrics_main`, the binary Waymo ships in the `waymo-open-dataset` repository,
running inside a `linux/amd64` container. Nothing in this project computes the number this prints;
`eval/to_waymo_objects.py` only repacks the tracker's CSV export and Waymo's own `lidar_box`
parquet into the two `Objects` proto files the binary reads.

```
./build/tracker --segment PATH --headless --sigma-position 0.036 --sigma-yaw 0.0088 \
    --seed 1 --export TRACKS.csv
eval/run_official.sh SEGMENT_NAME TRACKS.csv
```

`SEGMENT_NAME` is the full segment name (the same name the staged log and the parquet files carry).
The first run builds the evaluator image, which clones `waymo-open-dataset` and compiles only
`//waymo_open_dataset/metrics/tools:compute_tracking_metrics_main` with bazel inside a
`gcr.io/bazel-public/bazel:6.4.0` container; later runs reuse the built image. The script converts
the tracks CSV and the ground truth into a temporary directory, runs the container against both
files, and prints the binary's own output verbatim - a MOTA, MOTP, miss, mismatch and false-positive
breakdown per object type, per range bucket and per difficulty level.

The Python bindings the converter imports live in `eval/generated/`, generated once from the
cloned repository's `.proto` files with the local `protoc` and not committed:

```
protoc -I ~/waymo-data/waymo-open-dataset/src --python_out=eval/generated \
    waymo_open_dataset/protos/metrics.proto waymo_open_dataset/dataset.proto \
    waymo_open_dataset/label.proto waymo_open_dataset/protos/breakdown.proto \
    waymo_open_dataset/protos/keypoint.proto waymo_open_dataset/protos/vector.proto \
    waymo_open_dataset/protos/map.proto
```

## Cross-checking with motmetrics

`eval/score_external.py` scores a tracker export against the same `lidar_box` ground truth using
only third-party code: `shapely` computes the box overlaps and `motmetrics` runs the CLEAR MOT
accumulation and metric formulas. It is a second, independent check on `metrics.cpp` and Waymo's
own evaluator, never the source of a reported number.

```
~/venvs/tracker/bin/pip install "motmetrics==1.4.0" shapely
```

```
~/venvs/tracker/bin/python eval/score_external.py \
    --segment 10203656353524179475_7625_000_7645_000 --tracks TRACKS.csv --class vehicle
```
