"""Reads Waymo's lidar_box parquet for one segment into GroundTruthBox.

Rows are unfiltered; the stager and both eval scripts apply their own filter.
"""

import hashlib
from typing import NamedTuple, Optional

import pyarrow.parquet as pq

# Waymo's object type codes for the three tracked classes.
VEHICLE_CLASS = 1
PEDESTRIAN_CLASS = 2
CYCLIST_CLASS = 4
TRACKED_OBJECT_CLASSES = (VEHICLE_CLASS, PEDESTRIAN_CLASS, CYCLIST_CLASS)


class Box(NamedTuple):
    """An oriented 3D box with centre and size in metres and yaw in radians.
    Field order matches struct Box in src/types.hpp and the staged log."""
    center_x: float
    center_y: float
    center_z: float
    length: float
    width: float
    height: float
    yaw: float


class GroundTruthBox(NamedTuple):
    """One row of Waymo's lidar_box table, a labelled box in the vehicle
    frame of its capture time in microseconds."""
    capture_time_micros: int
    laser_object_id: str
    object_class: int
    box: Box
    num_lidar_points_in_box: int
    tracking_difficulty: Optional[int]


def stable_object_id(laser_object_id):
    """Returns the first 8 bytes of the SHA-1 of laser_object_id as a uint64,
    the identity the staged log and the motmetrics scorer share."""
    digest = hashlib.sha1(laser_object_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder = "big")


# The lidar_box columns to read, in GroundTruthBox field order; the seven
# box columns run in Box field order.
LIDAR_BOX_COLUMNS = (
    "key.frame_timestamp_micros",
    "key.laser_object_id",
    "[LiDARBoxComponent].type",
    "[LiDARBoxComponent].box.center.x",
    "[LiDARBoxComponent].box.center.y",
    "[LiDARBoxComponent].box.center.z",
    "[LiDARBoxComponent].box.size.x",
    "[LiDARBoxComponent].box.size.y",
    "[LiDARBoxComponent].box.size.z",
    "[LiDARBoxComponent].box.heading",
    "[LiDARBoxComponent].num_lidar_points_in_box",
    "[LiDARBoxComponent].difficulty_level.tracking",
)


def read_ground_truth_boxes(parquet_root, segment_name):
    """Returns one GroundTruthBox per row of the segment's lidar_box table,
    unfiltered.

    Boxes are in the vehicle frame of their own capture time, with centre and
    size in metres, yaw in radians and capture_time_micros in microseconds.
    """
    box_table = pq.read_table(
        f"{parquet_root}/lidar_box/{segment_name}.parquet")

    columns = [box_table.column(name).to_pylist() for name in LIDAR_BOX_COLUMNS]

    boxes = []
    for (
            capture_time_micros,
            laser_object_id,
            object_class,
            *box_values,
            num_lidar_points_in_box,
            tracking_difficulty,
    ) in zip(*columns):
        boxes.append(
            GroundTruthBox(
                capture_time_micros,
                laser_object_id,
                object_class,
                Box(*box_values),
                num_lidar_points_in_box,
                tracking_difficulty,
            ))
    return boxes
