"""
This script turns Waymo's own v2 parquet files for one segment into the binary log the tracker
reads. It joins the lidar range image, its calibration, the per-object ground truth boxes and the
vehicle pose by frame timestamp, converts the range image into vehicle-frame points, and writes the
result in the exact byte layout README.md describes. Nothing downstream of the log format depends
on this file - only this script and the C++ reader in log.cpp agree on what the bytes mean, and the
sanity check below exists to catch a wrong azimuth convention before it silently corrupts every
point in the segment.
"""

import argparse
import struct

import numpy as np
import pyarrow.parquet as pq

from waymo_boxes import read_labelled_boxes, stable_object_id

MAGIC = b"TRKLOG02"
VEHICLE_TYPE = 1
PEDESTRIAN_TYPE = 2
CYCLIST_TYPE = 4


def wrap_angle(radians):
    while radians > np.pi:
        radians -= 2.0 * np.pi
    while radians < -np.pi:
        radians += 2.0 * np.pi
    return radians


def read_components(parquet_root, segment, laser):
    lidar_table = pq.read_table(f"{parquet_root}/lidar/{segment}.parquet")
    calibration_table = pq.read_table(f"{parquet_root}/lidar_calibration/{segment}.parquet")
    pose_table = pq.read_table(f"{parquet_root}/vehicle_pose/{segment}.parquet")

    calibration_by_laser = {}
    for laser_name, extrinsic, inclination_min, inclination_max, inclination_values in zip(
        calibration_table.column("key.laser_name").to_pylist(),
        calibration_table.column("[LiDARCalibrationComponent].extrinsic.transform").to_pylist(),
        calibration_table.column("[LiDARCalibrationComponent].beam_inclination.min").to_pylist(),
        calibration_table.column("[LiDARCalibrationComponent].beam_inclination.max").to_pylist(),
        calibration_table.column("[LiDARCalibrationComponent].beam_inclination.values").to_pylist(),
    ):
        calibration_by_laser[laser_name] = {
            "extrinsic": np.array(extrinsic, dtype=np.float64).reshape(4, 4),
            "inclination_min": inclination_min,
            "inclination_max": inclination_max,
            "inclination_values": np.array(inclination_values, dtype=np.float64) if inclination_values else None,
        }
    calibration = calibration_by_laser[laser]

    points_by_frame = {}
    for frame_timestamp, laser_name, range_image_values, range_image_shape in zip(
        lidar_table.column("key.frame_timestamp_micros").to_pylist(),
        lidar_table.column("key.laser_name").to_pylist(),
        lidar_table.column("[LiDARComponent].range_image_return1.values").to_pylist(),
        lidar_table.column("[LiDARComponent].range_image_return1.shape").to_pylist(),
    ):
        if laser_name != laser:
            continue
        points_by_frame[frame_timestamp] = range_image_to_points(range_image_values, range_image_shape, calibration)

    ground_truth_by_frame = {}
    for box in read_labelled_boxes(parquet_root, segment):
        if box["object_type"] not in (VEHICLE_TYPE, PEDESTRIAN_TYPE, CYCLIST_TYPE):
            continue
        ground_truth_by_frame.setdefault(box["frame_timestamp_micros"], []).append({
            "object_id": stable_object_id(box["laser_object_id"]),
            "object_class": box["object_type"],
            "center_x": box["center_x"],
            "center_y": box["center_y"],
            "center_z": box["center_z"],
            "length": box["length"],
            "width": box["width"],
            "height": box["height"],
            "heading": box["heading"],
            "num_lidar_points_in_box": box["num_lidar_points_in_box"],
        })

    pose_by_frame = dict(zip(
        pose_table.column("key.frame_timestamp_micros").to_pylist(),
        pose_table.column("[VehiclePoseComponent].world_from_vehicle.transform").to_pylist(),
    ))

    frames = []
    for frame_timestamp in sorted(points_by_frame):
        if frame_timestamp not in pose_by_frame:
            continue
        frames.append({
            "capture_time_micros": frame_timestamp,
            "vehicle_to_world": np.array(pose_by_frame[frame_timestamp], dtype=np.float64).reshape(4, 4),
            "points": points_by_frame[frame_timestamp],
            "ground_truth": ground_truth_by_frame.get(frame_timestamp, []),
        })
    return frames


def range_image_to_points(range_image_values, range_image_shape, calibration):
    height, width, channel_count = range_image_shape
    range_image = np.array(range_image_values, dtype=np.float64).reshape(height, width, channel_count)
    range_channel = range_image[:, :, 0]

    inclination_values = calibration["inclination_values"]
    if inclination_values is not None and len(inclination_values) == height:
        inclinations = inclination_values[::-1]
    else:
        inclinations = np.linspace(calibration["inclination_max"], calibration["inclination_min"], height)

    extrinsic = calibration["extrinsic"]
    azimuth_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    column_indices = np.arange(width)
    azimuths = np.pi - (column_indices + 0.5) * 2.0 * np.pi / width - azimuth_correction

    inclination_grid, azimuth_grid = np.meshgrid(inclinations, azimuths, indexing="ij")

    valid_mask = range_channel > 0.0
    ranges = range_channel[valid_mask]
    valid_inclinations = inclination_grid[valid_mask]
    valid_azimuths = azimuth_grid[valid_mask]

    sensor_x = ranges * np.cos(valid_inclinations) * np.cos(valid_azimuths)
    sensor_y = ranges * np.cos(valid_inclinations) * np.sin(valid_azimuths)
    sensor_z = ranges * np.sin(valid_inclinations)

    sensor_points = np.stack([sensor_x, sensor_y, sensor_z, np.ones_like(sensor_x)], axis=1)
    vehicle_points = (extrinsic @ sensor_points.T).T

    points = np.empty((vehicle_points.shape[0], 3), dtype=np.float32)
    points[:, 0:3] = vehicle_points[:, 0:3]
    return points


def count_points_in_box(points, box):
    cosine = np.cos(-box["heading"])
    sine = np.sin(-box["heading"])
    relative_x = points[:, 0] - box["center_x"]
    relative_y = points[:, 1] - box["center_y"]
    local_x = cosine * relative_x - sine * relative_y
    local_y = sine * relative_x + cosine * relative_y
    inside_footprint = (np.abs(local_x) <= box["length"] / 2.0) & (np.abs(local_y) <= box["width"] / 2.0)
    inside_height = np.abs(points[:, 2] - box["center_z"]) <= box["height"] / 2.0
    return int(np.count_nonzero(inside_footprint & inside_height))


def measure_box_jitter(boxes_by_frame):
    positions_by_object = {}
    for frame_index, boxes in enumerate(boxes_by_frame):
        for box in boxes:
            positions_by_object.setdefault(box["object_id"], []).append(
                (frame_index, box["center_x"], box["center_y"], box["heading"])
            )

    position_residuals = []
    heading_residuals = []
    for trajectory in positions_by_object.values():
        for index in range(1, len(trajectory) - 1):
            previous_frame, previous_x, previous_y, previous_heading = trajectory[index - 1]
            current_frame, current_x, current_y, current_heading = trajectory[index]
            next_frame, next_x, next_y, next_heading = trajectory[index + 1]
            if current_frame - previous_frame != 1 or next_frame - current_frame != 1:
                continue
            position_residuals.append(next_x - (2.0 * current_x - previous_x))
            position_residuals.append(next_y - (2.0 * current_y - previous_y))
            heading_residuals.append(wrap_angle(next_heading - (2.0 * current_heading - previous_heading)))

    position_sigma = float(np.std(position_residuals)) if position_residuals else 0.0
    heading_sigma = float(np.std(heading_residuals)) if heading_residuals else 0.0
    print(f"measured sigma_measurement_position: {position_sigma:.6f} m")
    print(f"measured sigma_measurement_yaw: {heading_sigma:.6f} rad")
    return position_sigma, heading_sigma


def write_log(path, segment_name, frames):
    with open(path, "wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(frames)))
        name_bytes = segment_name.encode("utf-8")
        stream.write(struct.pack("<I", len(name_bytes)))
        stream.write(name_bytes)
        for frame in frames:
            stream.write(struct.pack("<q", frame["capture_time_micros"]))
            stream.write(struct.pack("<16d", *frame["vehicle_to_world"].flatten().tolist()))
            points = frame["points"]
            stream.write(struct.pack("<I", len(points)))
            stream.write(points.astype("<f4").tobytes())
            ground_truth = frame["ground_truth"]
            stream.write(struct.pack("<I", len(ground_truth)))
            for box in ground_truth:
                stream.write(struct.pack("<Q", box["object_id"]))
                stream.write(struct.pack("<B", box["object_class"]))
                stream.write(struct.pack(
                    "<7d", box["center_x"], box["center_y"], box["center_z"],
                    box["length"], box["width"], box["height"], box["heading"],
                ))
                stream.write(struct.pack("<i", box["num_lidar_points_in_box"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--laser", type=int, default=1)
    arguments = parser.parse_args()

    frames = read_components(arguments.parquet_root, arguments.segment, arguments.laser)

    if frames:
        first_frame = frames[0]
        staged_points_in_boxes = sum(count_points_in_box(first_frame["points"], box) for box in first_frame["ground_truth"])
        reported_points_in_boxes = sum(box["num_lidar_points_in_box"] for box in first_frame["ground_truth"])
        ratio = staged_points_in_boxes / reported_points_in_boxes if reported_points_in_boxes else 0.0
        print(f"staged/reported point ratio for first frame: {ratio:.4f}")
        if not 0.5 <= ratio <= 1.5:
            print("WARNING: point ratio far from 1 - the azimuth convention is likely wrong")

    measure_box_jitter([frame["ground_truth"] for frame in frames])

    write_log(arguments.out, arguments.segment, frames)


if __name__ == "__main__":
    main()
