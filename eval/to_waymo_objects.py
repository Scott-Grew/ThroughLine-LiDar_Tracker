"""
This script turns the tracker's CSV export and Waymo's own lidar_box parquet into the two binary
files compute_tracking_metrics_main reads: one serialized Objects proto of predictions, one of
ground truth. It computes no metric and makes no matching decision - it only repacks fields that
already exist, in the units and identities they already carry, into the proto Waymo's evaluator
expects. The evaluator itself is the only thing in this project allowed to produce a score.

Both inputs are already in the per-frame vehicle frame: export.cpp converts every confirmed track
out of the tracker's world frame and back into the vehicle frame of the frame it was seen in before
writing the CSV, and the lidar_box parquet columns read below are Waymo's own box coordinates,
published in that same per-frame vehicle frame. score_external.py already relies on this and notes
it directly; nothing here needs to transform a coordinate.

Object identity differs by file on purpose. A prediction's identity is this project's own track id,
so it is written as the decimal string of the tracker's uint64. A ground truth object's identity is
Waymo's own label id string, key.laser_object_id, written back unchanged - unlike
stage_segment.py's staged log and score_external.py's motmetrics accumulator, neither of which can
hold a string id and so both hash it into a number first. Waymo's own Label.id field is a string,
so that hashing step has no reason to happen here.

compute_tracking_metrics_main.cc reads ground truth tracking difficulty straight from each object's
tracking_difficulty_level field - it does no auto-computation from num_lidar_points_in_box the way
compute_detection_metrics_main.cc does for detection difficulty. That field is still written here,
both because it is already on hand and because it is this project's own eligibility rule, shared
with score_external.py and the staged log: a labelled box the sensor returned no points for is not
a scorable object.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "generated"))
from waymo_open_dataset.protos import metrics_pb2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "stage"))
from waymo_boxes import read_labelled_boxes

TRACKED_WAYMO_TYPES = {1, 2, 4}


# Builds one ground truth Objects proto from Waymo's own lidar_box parquet for one segment, keeping
# only vehicle, pedestrian and cyclist boxes the sensor returned at least one point for - the same
# subset this project's tracker is ever asked to track. Every object carries the segment's context
# name and its own frame timestamp so compute_tracking_metrics_main can group objects back into
# frames, and its identity is Waymo's own laser_object_id string, unchanged.
def build_ground_truth_objects(parquet_root, segment):
    objects = metrics_pb2.Objects()
    for box in read_labelled_boxes(parquet_root, segment):
        if box["object_type"] not in TRACKED_WAYMO_TYPES or box["num_lidar_points_in_box"] <= 0:
            continue

        waymo_object = objects.objects.add()
        waymo_object.context_name = segment
        waymo_object.frame_timestamp_micros = box["frame_timestamp_micros"]
        waymo_object.object.id = box["laser_object_id"]
        waymo_object.object.type = box["object_type"]
        waymo_object.object.num_lidar_points_in_box = box["num_lidar_points_in_box"]
        waymo_object.object.tracking_difficulty_level = box["tracking_difficulty"] if box["tracking_difficulty"] is not None else 0
        waymo_object.object.box.center_x = box["center_x"]
        waymo_object.object.box.center_y = box["center_y"]
        waymo_object.object.box.center_z = box["center_z"]
        waymo_object.object.box.length = box["length"]
        waymo_object.object.box.width = box["width"]
        waymo_object.object.box.height = box["height"]
        waymo_object.object.box.heading = box["heading"]

    return objects


# Builds one predictions Objects proto from the tracker's CSV export. Every row becomes one Object
# with score fixed at 1.0, since this tracker never expresses detection confidence, and identity
# set to the decimal string of its track id - the tracking identity compute_tracking_metrics_main
# uses to decide whether a track kept the same id across frames.
def build_prediction_objects(tracks_path, segment):
    tracks_table = pd.read_csv(tracks_path)

    objects = metrics_pb2.Objects()
    for row in tracks_table.itertuples(index=False):
        waymo_object = objects.objects.add()
        waymo_object.context_name = segment
        waymo_object.frame_timestamp_micros = row.frame_timestamp_micros
        waymo_object.object.id = str(row.track_id)
        waymo_object.object.type = row.object_class
        waymo_object.score = 1.0
        waymo_object.object.box.center_x = row.center_x
        waymo_object.object.box.center_y = row.center_y
        waymo_object.object.box.center_z = row.center_z
        waymo_object.object.box.length = row.length
        waymo_object.object.box.width = row.width
        waymo_object.object.box.height = row.height
        waymo_object.object.box.heading = row.yaw

    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out-predictions", required=True)
    parser.add_argument("--out-ground-truth", required=True)
    arguments = parser.parse_args()

    prediction_objects = build_prediction_objects(arguments.tracks, arguments.segment)
    ground_truth_objects = build_ground_truth_objects(arguments.parquet_root, arguments.segment)

    with open(arguments.out_predictions, "wb") as stream:
        stream.write(prediction_objects.SerializeToString())
    with open(arguments.out_ground_truth, "wb") as stream:
        stream.write(ground_truth_objects.SerializeToString())

    print(f"wrote {len(prediction_objects.objects)} prediction objects to {arguments.out_predictions}")
    print(f"wrote {len(ground_truth_objects.objects)} ground truth objects to {arguments.out_ground_truth}")


if __name__ == "__main__":
    main()
