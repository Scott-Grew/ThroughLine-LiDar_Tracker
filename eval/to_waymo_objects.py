"""Packs the tracker's CSV export and Waymo's lidar_box parquet into the two
Objects protos that Waymo's compute_tracking_metrics_main reads."""

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "generated"))
from waymo_open_dataset.protos import metrics_pb2

sys.path.insert(0,
                str(pathlib.Path(__file__).resolve().parent.parent / "stage"))
from waymo_boxes import read_ground_truth_boxes, TRACKED_OBJECT_CLASSES


def build_ground_truth_objects(parquet_root, segment_name):
    """Returns an Objects proto of the tracked classes' boxes that have at
    least one lidar point; the object id stays Waymo's laser_object_id."""
    objects = metrics_pb2.Objects()
    for ground_truth_box in read_ground_truth_boxes(parquet_root, segment_name):
        if (ground_truth_box.object_class not in TRACKED_OBJECT_CLASSES or
                ground_truth_box.num_lidar_points_in_box <= 0):
            continue

        waymo_object = objects.objects.add()
        waymo_object.context_name = segment_name
        waymo_object.frame_timestamp_micros = (
            ground_truth_box.capture_time_micros)
        waymo_object.object.id = ground_truth_box.laser_object_id
        waymo_object.object.type = ground_truth_box.object_class
        waymo_object.object.num_lidar_points_in_box = (
            ground_truth_box.num_lidar_points_in_box)
        waymo_object.object.tracking_difficulty_level = (
            ground_truth_box.tracking_difficulty
            if ground_truth_box.tracking_difficulty is not None else 0)
        waymo_object.object.box.center_x = ground_truth_box.box.center_x
        waymo_object.object.box.center_y = ground_truth_box.box.center_y
        waymo_object.object.box.center_z = ground_truth_box.box.center_z
        waymo_object.object.box.length = ground_truth_box.box.length
        waymo_object.object.box.width = ground_truth_box.box.width
        waymo_object.object.box.height = ground_truth_box.box.height
        waymo_object.object.box.heading = ground_truth_box.box.yaw

    return objects


def build_prediction_objects(tracks_path, segment_name):
    """Returns an Objects proto of the CSV export's rows; the score is fixed at
    1.0 and the object id is the track id as a string."""
    tracks_table = pd.read_csv(tracks_path)

    objects = metrics_pb2.Objects()
    for row in tracks_table.itertuples(index = False):
        waymo_object = objects.objects.add()
        waymo_object.context_name = segment_name
        waymo_object.frame_timestamp_micros = (row.frame_timestamp_micros)
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
    parser.add_argument("--parquet-root", required = True)
    parser.add_argument("--segment", dest = "segment_name", required = True)
    parser.add_argument("--tracks", required = True)
    parser.add_argument("--out-predictions", required = True)
    parser.add_argument("--out-ground-truth", required = True)
    arguments = parser.parse_args()

    prediction_objects = build_prediction_objects(arguments.tracks,
                                                  arguments.segment_name)
    ground_truth_objects = build_ground_truth_objects(arguments.parquet_root,
                                                      arguments.segment_name)

    with open(arguments.out_predictions, "wb") as stream:
        stream.write(prediction_objects.SerializeToString())
    with open(arguments.out_ground_truth, "wb") as stream:
        stream.write(ground_truth_objects.SerializeToString())

    print(f"wrote {len(prediction_objects.objects)} prediction objects "
          f"to {arguments.out_predictions}")
    print(f"wrote {len(ground_truth_objects.objects)} ground truth objects "
          f"to {arguments.out_ground_truth}")


if __name__ == "__main__":
    main()
