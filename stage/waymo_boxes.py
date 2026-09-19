# Reads Waymo's lidar_box parquet for one segment into one plain
# dict per row, unfiltered; every caller applies its own filter.

import hashlib

import pyarrow.parquet as pq


# Hashes a laser_object_id string into the uint64 identity the
# staged log and the motmetrics accumulator both use.
def stable_object_id(laser_object_id):
    digest = hashlib.sha1(laser_object_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


# Reads every row of one segment's lidar_box table into a dict,
# applying no class or lidar-point-count filter.
def read_labelled_boxes(parquet_root, segment_name):
    box_table = pq.read_table(
        f"{parquet_root}/lidar_box/{segment_name}.parquet"
    )

    boxes = []
    for (
        frame_timestamp_micros,
        laser_object_id,
        object_type,
        center_x,
        center_y,
        center_z,
        length,
        width,
        height,
        heading,
        num_lidar_points_in_box,
        tracking_difficulty,
    ) in zip(
        box_table.column("key.frame_timestamp_micros").to_pylist(),
        box_table.column("key.laser_object_id").to_pylist(),
        box_table.column("[LiDARBoxComponent].type").to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].box.center.x"
        ).to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].box.center.y"
        ).to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].box.center.z"
        ).to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].box.size.x"
        ).to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].box.size.y"
        ).to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].box.size.z"
        ).to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].box.heading"
        ).to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].num_lidar_points_in_box"
        ).to_pylist(),
        box_table.column(
            "[LiDARBoxComponent].difficulty_level.tracking"
        ).to_pylist(),
    ):
        boxes.append(
            {
                "frame_timestamp_micros": frame_timestamp_micros,
                "laser_object_id": laser_object_id,
                "object_type": object_type,
                "center_x": center_x,
                "center_y": center_y,
                "center_z": center_z,
                "length": length,
                "width": width,
                "height": height,
                "heading": heading,
                "num_lidar_points_in_box": num_lidar_points_in_box,
                "tracking_difficulty": tracking_difficulty,
            }
        )
    return boxes
