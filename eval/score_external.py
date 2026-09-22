"""Development monitor that scores a tracker export against Waymo's ground
truth.

Matching uses shapely for 3D IoU and motmetrics for the CLEAR MOT numbers;
the MOTP it prints is 1 - IoU.
"""

import argparse
import pathlib
import sys
from typing import NamedTuple

import motmetrics as mm
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon

sys.path.insert(0,
                str(pathlib.Path(__file__).resolve().parent.parent / "stage"))
from waymo_boxes import (read_ground_truth_boxes, stable_object_id, Box,
                         VEHICLE_CLASS, PEDESTRIAN_CLASS, CYCLIST_CLASS)

# Maps the --class flag's names to Waymo's object type codes.
CLASS_NAME_TO_OBJECT_CLASS = {
    "vehicle": VEHICLE_CLASS,
    "pedestrian": PEDESTRIAN_CLASS,
    "cyclist": CYCLIST_CLASS,
}

# Waymo's vehicle IoU threshold; its pedestrian and cyclist threshold is 0.5.
DEFAULT_IOU_THRESHOLD = 0.7


class IdentifiedBox(NamedTuple):
    """A box with the identity motmetrics matches on, the hashed Waymo
    object id for ground truth and the track id for a prediction."""
    identity: int
    box: Box


def footprint_polygon(box):
    """Returns the box's bird's-eye footprint as a shapely Polygon, with
    centre, length and width in metres and yaw in radians."""
    half_length = box.length / 2.0
    half_width = box.width / 2.0
    local_corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    cosine = np.cos(box.yaw)
    sine = np.sin(box.yaw)
    # Each corner (x, y) rotates by yaw to (x cos - y sin, x sin + y cos)
    # and then shifts to the box centre.
    world_corners = [(
        box.center_x + cosine * local_x - sine * local_y,
        box.center_y + sine * local_x + cosine * local_y,
    ) for local_x, local_y in local_corners]
    return Polygon(world_corners)


def intersection_over_union_3d(first_box, second_box):
    """Returns the 3D IoU of two Box in the same frame, the footprint overlap
    area times the vertical overlap over the union volume."""
    first_footprint = footprint_polygon(first_box)
    second_footprint = footprint_polygon(second_box)
    footprint_intersection_area = first_footprint.intersection(
        second_footprint).area
    if footprint_intersection_area <= 0.0:
        return 0.0

    first_z_min, first_z_max = (
        first_box.center_z - first_box.height / 2.0,
        first_box.center_z + first_box.height / 2.0,
    )
    second_z_min, second_z_max = (
        second_box.center_z - second_box.height / 2.0,
        second_box.center_z + second_box.height / 2.0,
    )
    vertical_overlap = max(
        0.0,
        min(first_z_max, second_z_max) - max(first_z_min, second_z_min),
    )
    if vertical_overlap <= 0.0:
        return 0.0

    intersection_volume = (footprint_intersection_area * vertical_overlap)
    first_volume = (first_box.length * first_box.width * first_box.height)
    second_volume = (second_box.length * second_box.width * second_box.height)
    union_volume = first_volume + second_volume - intersection_volume
    if union_volume <= 0.0:
        return 0.0
    return intersection_volume / union_volume


def read_ground_truth(parquet_root, segment_name, class_name):
    """Returns a dict from capture time in microseconds to that frame's
    IdentifiedBox for one class, keeping boxes with at least one lidar
    point."""
    object_class = CLASS_NAME_TO_OBJECT_CLASS[class_name]

    ground_truth_by_frame = {}
    for ground_truth_box in read_ground_truth_boxes(parquet_root, segment_name):
        if (ground_truth_box.object_class != object_class or
                ground_truth_box.num_lidar_points_in_box <= 0):
            continue
        ground_truth_by_frame.setdefault(
            ground_truth_box.capture_time_micros, []).append(
                IdentifiedBox(
                    stable_object_id(ground_truth_box.laser_object_id),
                    ground_truth_box.box))
    return ground_truth_by_frame


def read_predictions(tracks_path, class_name):
    """Returns a dict from capture time in microseconds to that frame's
    IdentifiedBox for one class, read from the tracker's CSV export.

    The export is already in the per-frame vehicle frame the ground truth uses.
    """
    object_class = CLASS_NAME_TO_OBJECT_CLASS[class_name]
    tracks_table = pd.read_csv(tracks_path)
    tracks_table = tracks_table[tracks_table["object_class"] == object_class]

    predictions_by_frame = {}
    for row in tracks_table.itertuples(index = False):
        predictions_by_frame.setdefault(row.frame_timestamp_micros, []).append(
            IdentifiedBox(
                row.track_id,
                Box(
                    row.center_x,
                    row.center_y,
                    row.center_z,
                    row.length,
                    row.width,
                    row.height,
                    row.yaw,
                )))
    return predictions_by_frame


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required = True)
    parser.add_argument("--segment", dest = "segment_name", required = True)
    parser.add_argument("--tracks", required = True)
    parser.add_argument(
        "--class",
        dest = "class_name",
        choices = ["vehicle", "pedestrian", "cyclist"],
        default = "vehicle",
    )
    parser.add_argument(
        "--iou-threshold",
        type = float,
        default = DEFAULT_IOU_THRESHOLD,
        help = "3D IoU needed for a match; Waymo uses 0.7 for vehicles, "
        "0.5 otherwise")
    return parser.parse_args()


def iou_distance_matrix(ground_truth_boxes, prediction_boxes, iou_threshold):
    """Returns a (num_ground_truth, num_predictions) float array of
    1 - IoU, with np.nan where the IoU is below iou_threshold."""
    distance_matrix = np.full((len(ground_truth_boxes), len(prediction_boxes)),
                              np.nan)
    for ground_truth_index, ground_truth_box in enumerate(ground_truth_boxes):
        for prediction_index, prediction_box in enumerate(prediction_boxes):
            iou = intersection_over_union_3d(ground_truth_box.box,
                                             prediction_box.box)
            if iou >= iou_threshold:
                distance_matrix[ground_truth_index,
                                prediction_index] = (1.0 - iou)
    return distance_matrix


def accumulate_frames(ground_truth_by_frame, predictions_by_frame,
                      iou_threshold):
    """Returns an mm.MOTAccumulator updated with each frame's
    ground-truth ids, prediction ids and IoU distance matrix."""
    accumulator = mm.MOTAccumulator(auto_id = False)
    for capture_time_micros in sorted(
            set(ground_truth_by_frame) | set(predictions_by_frame)):
        ground_truth_boxes = ground_truth_by_frame.get(capture_time_micros, [])
        prediction_boxes = predictions_by_frame.get(capture_time_micros, [])

        ground_truth_ids = [box.identity for box in ground_truth_boxes]
        prediction_ids = [box.identity for box in prediction_boxes]

        distance_matrix = iou_distance_matrix(ground_truth_boxes,
                                              prediction_boxes, iou_threshold)

        accumulator.update(
            ground_truth_ids,
            prediction_ids,
            distance_matrix,
            frameid = capture_time_micros,
        )
    return accumulator


def main():
    arguments = parse_arguments()

    ground_truth_by_frame = read_ground_truth(
        arguments.parquet_root,
        arguments.segment_name,
        arguments.class_name,
    )
    predictions_by_frame = read_predictions(arguments.tracks,
                                            arguments.class_name)

    accumulator = accumulate_frames(ground_truth_by_frame, predictions_by_frame,
                                    arguments.iou_threshold)

    metrics_host = mm.metrics.create()
    summary = metrics_host.compute(
        accumulator,
        metrics = [
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "num_switches",
            "num_misses",
            "num_false_positives",
            "num_objects",
        ],
        name = f"{arguments.segment_name}/{arguments.class_name}",
    )

    print(f"scored with motmetrics {mm.__version__} and shapely "
          f"{shapely.__version__}")
    print(mm.io.render_summary(summary))


if __name__ == "__main__":
    main()
