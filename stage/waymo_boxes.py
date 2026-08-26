"""
This file reads Waymo's own lidar_box parquet for one segment, once, in the shape every caller
that needs ground truth boxes agrees on: one plain dict per row, carrying every column the box
table has and no class or point-count filter applied. stage_segment.py, eval/score_external.py
and eval/to_waymo_objects.py each used to read this same parquet table themselves, each with its
own slightly different set of columns and its own filter baked into the read. This is the one
place that read happens now; every caller applies its own filter to the rows this returns.
"""

import hashlib

import pyarrow.parquet as pq


# Waymo's ground truth object ids are strings (key.laser_object_id); the staged log and the
# motmetrics accumulator both used by this project can only carry a number as an object's
# identity, so both hash the same string into the same uint64 through this function, kept in one
# place so the two paths can never drift into hashing the same object into two different numbers.
def stable_object_id(laser_object_id):
    digest = hashlib.sha1(laser_object_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


# Reads every row of one segment's lidar_box parquet table into a plain dict per row, applying no
# filter at all - not on object type, not on lidar point count. Every caller wants a different
# filter over the same rows, so filtering here would bake one caller's rule into every caller's
# data.
def read_labelled_boxes(parquet_root, segment_name):
    box_table = pq.read_table(f"{parquet_root}/lidar_box/{segment_name}.parquet")

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
        box_table.column("[LiDARBoxComponent].box.center.x").to_pylist(),
        box_table.column("[LiDARBoxComponent].box.center.y").to_pylist(),
        box_table.column("[LiDARBoxComponent].box.center.z").to_pylist(),
        box_table.column("[LiDARBoxComponent].box.size.x").to_pylist(),
        box_table.column("[LiDARBoxComponent].box.size.y").to_pylist(),
        box_table.column("[LiDARBoxComponent].box.size.z").to_pylist(),
        box_table.column("[LiDARBoxComponent].box.heading").to_pylist(),
        box_table.column("[LiDARBoxComponent].num_lidar_points_in_box").to_pylist(),
        box_table.column("[LiDARBoxComponent].difficulty_level.tracking").to_pylist(),
    ):
        boxes.append({
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
        })
    return boxes
