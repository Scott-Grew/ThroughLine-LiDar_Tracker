# Development monitor: scores a tracker export against Waymo's
# ground truth via shapely and motmetrics. Its MOTP is 1 - IoU.

import argparse
import pathlib
import sys

import motmetrics as mm
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parent.parent / "stage")
)
from waymo_boxes import read_labelled_boxes, stable_object_id

# Maps the --class flag's names to Waymo's object type codes.
CLASS_NAME_TO_WAYMO_TYPE = {
    "vehicle": 1,
    "pedestrian": 2,
    "cyclist": 4,
}


# Returns the box's bird's-eye footprint as a shapely Polygon
# from its four corners, centre, length/width and heading.
def footprint_polygon(center_x, center_y, length, width, heading):
    half_length = length / 2.0
    half_width = width / 2.0
    local_corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    cosine = np.cos(heading)
    sine = np.sin(heading)
    world_corners = [
        (
            center_x + cosine * local_x - sine * local_y,
            center_y + sine * local_x + cosine * local_y,
        )
        for local_x, local_y in local_corners
    ]
    return Polygon(world_corners)


# 3D IoU: footprint intersection area times vertical overlap,
# divided by the union volume; 0.0 when the boxes do not overlap.
def intersection_over_union_3d(first_box, second_box):
    first_footprint = footprint_polygon(
        first_box["center_x"],
        first_box["center_y"],
        first_box["length"],
        first_box["width"],
        first_box["heading"],
    )
    second_footprint = footprint_polygon(
        second_box["center_x"],
        second_box["center_y"],
        second_box["length"],
        second_box["width"],
        second_box["heading"],
    )
    footprint_intersection_area = first_footprint.intersection(
        second_footprint
    ).area
    if footprint_intersection_area <= 0.0:
        return 0.0

    first_z_min, first_z_max = (
        first_box["center_z"] - first_box["height"] / 2.0,
        first_box["center_z"] + first_box["height"] / 2.0,
    )
    second_z_min, second_z_max = (
        second_box["center_z"] - second_box["height"] / 2.0,
        second_box["center_z"] + second_box["height"] / 2.0,
    )
    vertical_overlap = max(
        0.0,
        min(first_z_max, second_z_max)
        - max(first_z_min, second_z_min),
    )
    if vertical_overlap <= 0.0:
        return 0.0

    intersection_volume = (
        footprint_intersection_area * vertical_overlap
    )
    first_volume = (
        first_box["length"] * first_box["width"] * first_box["height"]
    )
    second_volume = (
        second_box["length"]
        * second_box["width"]
        * second_box["height"]
    )
    union_volume = first_volume + second_volume - intersection_volume
    if union_volume <= 0.0:
        return 0.0
    return intersection_volume / union_volume


# Reads Waymo's lidar_box parquet for one class, keeping only
# boxes with a lidar point; ids are hashed via stable_object_id.
def read_ground_truth(parquet_root, segment, class_name):
    waymo_type = CLASS_NAME_TO_WAYMO_TYPE[class_name]

    ground_truth_by_frame = {}
    for box in read_labelled_boxes(parquet_root, segment):
        if (
            box["object_type"] != waymo_type
            or box["num_lidar_points_in_box"] <= 0
        ):
            continue
        ground_truth_by_frame.setdefault(
            box["frame_timestamp_micros"], []
        ).append(
            {
                "object_id": stable_object_id(box["laser_object_id"]),
                "center_x": box["center_x"],
                "center_y": box["center_y"],
                "center_z": box["center_z"],
                "length": box["length"],
                "width": box["width"],
                "height": box["height"],
                "heading": box["heading"],
            }
        )
    return ground_truth_by_frame


# Reads the tracker's CSV export for one class; already in the
# same per-frame vehicle frame as the ground truth boxes.
def read_predictions(tracks_path, class_name):
    waymo_type = CLASS_NAME_TO_WAYMO_TYPE[class_name]
    tracks_table = pd.read_csv(tracks_path)
    tracks_table = tracks_table[
        tracks_table["object_class"] == waymo_type
    ]

    predictions_by_frame = {}
    for row in tracks_table.itertuples(index=False):
        predictions_by_frame.setdefault(
            row.frame_timestamp_micros, []
        ).append(
            {
                "track_id": row.track_id,
                "center_x": row.center_x,
                "center_y": row.center_y,
                "center_z": row.center_z,
                "length": row.length,
                "width": row.width,
                "height": row.height,
                "heading": row.yaw,
            }
        )
    return predictions_by_frame


# CLI entry point: accumulates per-frame matches into motmetrics
# and prints the CLEAR MOT summary.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument(
        "--class",
        dest="class_name",
        choices=["vehicle", "pedestrian", "cyclist"],
        default="vehicle",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    arguments = parser.parse_args()

    ground_truth_by_frame = read_ground_truth(
        arguments.parquet_root,
        arguments.segment,
        arguments.class_name,
    )
    predictions_by_frame = read_predictions(
        arguments.tracks, arguments.class_name
    )

    accumulator = mm.MOTAccumulator(auto_id=False)
    for frame_timestamp in sorted(
        set(ground_truth_by_frame) | set(predictions_by_frame)
    ):
        ground_truth_boxes = ground_truth_by_frame.get(
            frame_timestamp, []
        )
        prediction_boxes = predictions_by_frame.get(
            frame_timestamp, []
        )

        ground_truth_ids = [
            box["object_id"] for box in ground_truth_boxes
        ]
        prediction_ids = [box["track_id"] for box in prediction_boxes]

        distance_matrix = np.full(
            (len(ground_truth_boxes), len(prediction_boxes)), np.nan
        )
        for ground_truth_index, ground_truth_box in enumerate(
            ground_truth_boxes
        ):
            for prediction_index, prediction_box in enumerate(
                prediction_boxes
            ):
                iou = intersection_over_union_3d(
                    ground_truth_box, prediction_box
                )
                if iou >= arguments.iou_threshold:
                    distance_matrix[
                        ground_truth_index, prediction_index
                    ] = (1.0 - iou)

        accumulator.update(
            ground_truth_ids,
            prediction_ids,
            distance_matrix,
            frameid=frame_timestamp,
        )

    metrics_host = mm.metrics.create()
    summary = metrics_host.compute(
        accumulator,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "num_switches",
            "num_misses",
            "num_false_positives",
            "num_objects",
        ],
        name=f"{arguments.segment}/{arguments.class_name}",
    )

    print(
        f"scored with motmetrics {mm.__version__} and shapely (third-party, not this project's own code)"
    )
    print(mm.io.render_summary(summary))


if __name__ == "__main__":
    main()
