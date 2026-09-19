"""
Packs the tracker's CSV export and Waymo's lidar_box parquet into
the two Objects protos compute_tracking_metrics_main reads.
"""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parent / "generated")
)
from waymo_open_dataset.protos import metrics_pb2

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parent.parent / "stage")
)
from waymo_boxes import read_labelled_boxes

TRACKED_WAYMO_TYPES = {1, 2, 4}


# Builds one ground truth Objects proto from Waymo's lidar_box
# parquet; identity stays Waymo's own laser_object_id string.
def build_ground_truth_objects(parquet_root, segment):
    objects = metrics_pb2.Objects()
    for box in read_labelled_boxes(parquet_root, segment):
        if (
            box["object_type"] not in TRACKED_WAYMO_TYPES
            or box["num_lidar_points_in_box"] <= 0
        ):
            continue

        waymo_object = objects.objects.add()
        waymo_object.context_name = segment
        waymo_object.frame_timestamp_micros = box[
            "frame_timestamp_micros"
        ]
        waymo_object.object.id = box["laser_object_id"]
        waymo_object.object.type = box["object_type"]
        waymo_object.object.num_lidar_points_in_box = box[
            "num_lidar_points_in_box"
        ]
        waymo_object.object.tracking_difficulty_level = (
            box["tracking_difficulty"]
            if box["tracking_difficulty"] is not None
            else 0
        )
        waymo_object.object.box.center_x = box["center_x"]
        waymo_object.object.box.center_y = box["center_y"]
        waymo_object.object.box.center_z = box["center_z"]
        waymo_object.object.box.length = box["length"]
        waymo_object.object.box.width = box["width"]
        waymo_object.object.box.height = box["height"]
        waymo_object.object.box.heading = box["heading"]

    return objects


# Builds one predictions Objects proto from the tracker's CSV
# export; score is fixed at 1.0, id is the track id as a string.
def build_prediction_objects(tracks_path, segment):
    tracks_table = pd.read_csv(tracks_path)

    objects = metrics_pb2.Objects()
    for row in tracks_table.itertuples(index=False):
        waymo_object = objects.objects.add()
        waymo_object.context_name = segment
        waymo_object.frame_timestamp_micros = (
            row.frame_timestamp_micros
        )
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


# CLI entry point: builds both protos and writes each to its own
# file.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out-predictions", required=True)
    parser.add_argument("--out-ground-truth", required=True)
    arguments = parser.parse_args()

    prediction_objects = build_prediction_objects(
        arguments.tracks, arguments.segment
    )
    ground_truth_objects = build_ground_truth_objects(
        arguments.parquet_root, arguments.segment
    )

    with open(arguments.out_predictions, "wb") as stream:
        stream.write(prediction_objects.SerializeToString())
    with open(arguments.out_ground_truth, "wb") as stream:
        stream.write(ground_truth_objects.SerializeToString())

    print(
        f"wrote {len(prediction_objects.objects)} prediction objects to {arguments.out_predictions}"
    )
    print(
        f"wrote {len(ground_truth_objects.objects)} ground truth objects to {arguments.out_ground_truth}"
    )


if __name__ == "__main__":
    main()
