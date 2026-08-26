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

This fetches Eigen 3.4.0, Catch2 v3.7.1, GLFW 3.4, Dear ImGui v1.91.5 and the
Rerun C++ SDK 0.26.0 via CMake FetchContent on first configure. The Rerun SDK
version is pinned at 0.26.0 rather than latest because Rerun stopped shipping
a macOS x86_64 prebuilt library after that release, and this is an Intel Mac.
Only the `tracker` executable links Rerun and Arrow; `tracker_core` and
`tracker_tests` stay free of both. Arrow is built from source inside `build/`
by the Rerun SDK's own FetchContent step (`RERUN_DOWNLOAD_AND_BUILD_ARROW`,
left at its default of ON) - nothing is installed system-wide, and this first
build is a one-off cost of a little over 16 minutes under `-j2` on this
machine.

The viewer application that the live `--rerun` flag spawns is a separate
program, not part of this build: `~/venvs/tracker/bin/pip install
rerun-sdk==0.26.0` installs it into the same virtual environment the stager
uses.

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
                [--sigma-position METRES] [--sigma-yaw RADIANS]
                [--seed N] [--rate R] [--headless] [--rerun] [--rerun-save PATH]
                [--export PATH] [--help]
```

`--sigma-position` and `--sigma-yaw` set the filter's assumed measurement
noise (defaults 0.1 m and 0.02 rad); the stager prints its own measured
values for a staged segment, which are the numbers to pass here.

`--headless` runs the tracker without the viewer and prints per-class MOTA,
MOTP, id switches, misses and false positives, plus the tracker step
p50/p99, overrun count and frame count. Without `--headless` the viewer
opens alongside the tracker running on its own thread: a small ImGui window
of perturbation sliders, and, when `--rerun` is also given, a live Rerun
window drawing the lidar points, the ego box, and every confirmed track's
box, history trail, predicted path and uncertainty ellipse. `--rerun-save
PATH` writes a gapless `.rrd` recording of the whole run afterward, built
from the same stored tracks the `--export` CSV comes from, whether or not
`--rerun` was given.

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
This is what a Python script in the container turns into Waymo's Objects
proto for the official evaluator; `eval/run_official.sh` stays a stub until
that wiring exists.

## Results

Scored by `motmetrics` 1.4.0, the MOTChallenge CLEAR implementation, against ground truth read
straight from Waymo's `lidar_box` parquet, with box overlap computed by `shapely` 2.1.2. No code
in this project takes part in the scoring.

Three validation segments from Waymo Open Dataset Perception v2, staged 2026-08-25. Detections
are the labelled boxes that the sensor returned at least one point for, so this measures the
tracker alone, with detection held perfect. It is not comparable to a leaderboard result, which
scores a detector and a tracker together.

Vehicle at IoU 0.7, pedestrian at IoU 0.5:

| segment | class | MOTA | IDF1 | switches | misses | false positives | objects |
|---|---|---|---|---|---|---|---|
| 10203656353524179475 | vehicle | 0.878 | 0.889 | 15 | 133 | 213 | 2946 |
| 1024360143612057520 | vehicle | 0.956 | 0.924 | 25 | 150 | 162 | 7645 |
| 10247954040621004675 | vehicle | 0.955 | 0.970 | 1 | 87 | 80 | 3730 |
| 10203656353524179475 | pedestrian | 0.783 | 0.724 | 19 | 59 | 106 | 847 |
| 1024360143612057520 | pedestrian | 0.911 | 0.788 | 35 | 97 | 200 | 3710 |
| 10247954040621004675 | pedestrian | 0.734 | 0.659 | 4 | 8 | 22 | 128 |

Tracker step time on these runs: 0.07 ms median, worst p99 0.632 ms over 36 runs, against the
100 ms a frame allows at 10 Hz.

`metrics.cpp`, this project's in-process monitor, agrees with motmetrics exactly on all six runs,
including the integer switch, miss and false-positive counts. It exists to watch a run as it
happens and is never the source of a reported number.

Waymo's own evaluator is not wired up yet, so nothing here is a Waymo benchmark result.

Reproducing a row:

```
~/venvs/tracker/bin/pip install "motmetrics==1.4.0" shapely
./build/tracker --segment PATH --headless --sigma-position 0.036 --sigma-yaw 0.0088 \
    --seed 1 --export TRACKS.csv
~/venvs/tracker/bin/python eval/score_external.py --parquet-root ROOT --segment SEGMENT_NAME \
    --tracks TRACKS.csv --class vehicle --iou-threshold 0.7
```

`--class` is `vehicle`, `pedestrian` or `cyclist`; `--iou-threshold` is 0.7 for vehicles and 0.5
for pedestrians and cyclists, matching `metrics.cpp`'s own per-class thresholds.

One convention differs on purpose: `motmetrics` reports MOTP as an average distance (`1 - IoU`,
lower is better), while `metrics.cpp` reports MOTP as an average IoU (higher is better). The two
are not the same quantity and this project does not convert between them.
