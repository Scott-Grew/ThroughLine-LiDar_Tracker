"""Stages one Waymo v2 parquet segment as the binary log the C++ tracker reads.

Only this file and src/log.cpp know the byte layout; change them together.
"""

import argparse
import struct

import numpy as np
import pyarrow.parquet as pq

from waymo_boxes import (read_ground_truth_boxes, stable_object_id,
                         TRACKED_OBJECT_CLASSES)

# File tag at the start of a staged log; must equal kMagic in src/log.cpp.
MAGIC = b"TRKLOG03"

# lidar_calibration columns, in read_calibration's loop unpacking order.
CALIBRATION_COLUMNS = (
    "key.laser_name",
    "[LiDARCalibrationComponent].extrinsic.transform",
    "[LiDARCalibrationComponent].beam_inclination.min",
    "[LiDARCalibrationComponent].beam_inclination.max",
    "[LiDARCalibrationComponent].beam_inclination.values",
)

# lidar columns, in read_points_by_frame's loop unpacking order.
LIDAR_COLUMNS = (
    "key.frame_timestamp_micros",
    "key.laser_name",
    "[LiDARComponent].range_image_return1.values",
    "[LiDARComponent].range_image_return1.shape",
)

# vehicle_pose columns, in read_pose_by_frame's zip unpacking order.
POSE_COLUMNS = (
    "key.frame_timestamp_micros",
    "[VehiclePoseComponent].world_from_vehicle.transform",
)

# Accepted range for staged points in boxes over Waymo's reported count.
MIN_POINT_RATIO = 0.5
MAX_POINT_RATIO = 1.5


def wrap_angle(radians):
    """Wraps an angle in radians into [-pi, pi]."""
    return float(np.angle(np.exp(1j * radians)))


def read_calibration(parquet_root, segment_name, laser):
    """Returns the requested laser's calibration dict, with a 4x4
    float64 extrinsic transform and its beam inclination bounds/values."""
    calibration_table = pq.read_table(
        f"{parquet_root}/lidar_calibration/{segment_name}.parquet")

    columns = [
        calibration_table.column(name).to_pylist()
        for name in CALIBRATION_COLUMNS
    ]

    calibration_by_laser = {}
    for (
            laser_name,
            extrinsic,
            inclination_min,
            inclination_max,
            inclination_values,
    ) in zip(*columns):
        calibration_by_laser[laser_name] = {
            "extrinsic": np.array(extrinsic, dtype = np.float64).reshape(4, 4),
            "inclination_min": inclination_min,
            "inclination_max": inclination_max,
            "inclination_values":
                (np.array(inclination_values, dtype = np.float64)
                 if inclination_values else None),
        }
    return calibration_by_laser[laser]


def read_points_by_frame(parquet_root, segment_name, laser, calibration):
    """Returns points_by_frame: a dict mapping frame timestamp in
    microseconds to that frame's (N, 3) float32 vehicle-frame points."""
    lidar_table = pq.read_table(f"{parquet_root}/lidar/{segment_name}.parquet")

    columns = [lidar_table.column(name).to_pylist() for name in LIDAR_COLUMNS]

    points_by_frame = {}
    for (
            capture_time_micros,
            laser_name,
            range_image_values,
            range_image_shape,
    ) in zip(*columns):
        if laser_name != laser:
            continue
        points_by_frame[capture_time_micros] = range_image_to_points(
            range_image_values, range_image_shape, calibration)
    return points_by_frame


def read_ground_truth_by_frame(parquet_root, segment_name):
    """Returns ground_truth_by_frame: a dict mapping frame timestamp
    in microseconds to a list of GroundTruthBox for that frame."""
    ground_truth_by_frame = {}
    for ground_truth_box in read_ground_truth_boxes(parquet_root, segment_name):
        if ground_truth_box.object_class not in TRACKED_OBJECT_CLASSES:
            continue
        ground_truth_by_frame.setdefault(ground_truth_box.capture_time_micros,
                                         []).append(ground_truth_box)
    return ground_truth_by_frame


def read_pose_by_frame(parquet_root, segment_name):
    """Returns pose_by_frame: a dict mapping frame timestamp in
    microseconds to a flattened 16-element world-from-vehicle transform."""
    pose_table = pq.read_table(
        f"{parquet_root}/vehicle_pose/{segment_name}.parquet")
    columns = [pose_table.column(name).to_pylist() for name in POSE_COLUMNS]
    return dict(zip(*columns))


def read_components(parquet_root, segment_name, laser):
    """Returns the segment's frames in time order, for one laser.

    Each frame is a dict: capture_time_micros, vehicle_to_world as a (4, 4)
    float64 array, points as (N, 3) float32 vehicle-frame metres, and
    ground_truth as a list of GroundTruthBox. Frames without a pose are
    skipped.
    """
    calibration = read_calibration(parquet_root, segment_name, laser)
    points_by_frame = read_points_by_frame(parquet_root, segment_name, laser,
                                           calibration)
    ground_truth_by_frame = read_ground_truth_by_frame(parquet_root,
                                                       segment_name)
    pose_by_frame = read_pose_by_frame(parquet_root, segment_name)

    frames = []
    for capture_time_micros in sorted(points_by_frame):
        if capture_time_micros not in pose_by_frame:
            continue
        frames.append({
            "capture_time_micros": capture_time_micros,
            "vehicle_to_world": np.array(pose_by_frame[capture_time_micros],
                                         dtype = np.float64).reshape(4, 4),
            "points": points_by_frame[capture_time_micros],
            "ground_truth": ground_truth_by_frame.get(capture_time_micros, []),
        })
    return frames


def range_image_to_points(range_image_values, range_image_shape, calibration):
    """Converts one laser's range image to (N, 3) float32 vehicle-frame metres.

    range_image_values is the flattened image and range_image_shape its
    (rows, columns, channels), (64, 2650, 4) for laser 1. Channel 0 is range
    in metres; cells without a return hold -1 there and are dropped.
    """
    height, width, channel_count = range_image_shape
    range_image = np.array(range_image_values,
                           dtype = np.float64).reshape(height, width,
                                                       channel_count)
    range_channel = range_image[:, :, 0]

    inclination_values = calibration["inclination_values"]
    if (inclination_values is not None and len(inclination_values) == height):
        # Calibration's inclination list runs opposite the image's
        # row order, hence the reversal.
        inclinations = inclination_values[::-1]
    else:
        inclinations = np.linspace(
            calibration["inclination_max"],
            calibration["inclination_min"],
            height,
        )

    extrinsic = calibration["extrinsic"]
    azimuth_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    column_indices = np.arange(width)
    # Azimuth convention here is easy to get backwards; check_point_ratio
    # exists to catch it.
    azimuths = (np.pi - (column_indices + 0.5) * 2.0 * np.pi / width -
                azimuth_correction)

    inclination_grid, azimuth_grid = np.meshgrid(inclinations,
                                                 azimuths,
                                                 indexing = "ij")

    has_return = range_channel > 0.0
    ranges = range_channel[has_return]
    return_inclinations = inclination_grid[has_return]
    return_azimuths = azimuth_grid[has_return]

    sensor_x = (ranges * np.cos(return_inclinations) * np.cos(return_azimuths))
    sensor_y = (ranges * np.cos(return_inclinations) * np.sin(return_azimuths))
    sensor_z = ranges * np.sin(return_inclinations)

    sensor_points = np.stack(
        [sensor_x, sensor_y, sensor_z,
         np.ones_like(sensor_x)], axis = 1)
    vehicle_points = (extrinsic @ sensor_points.T).T

    points = np.empty((vehicle_points.shape[0], 3), dtype = np.float32)
    points[:, 0:3] = vehicle_points[:, 0:3]
    return points


def count_points_in_box(points, box):
    """Counts the (N, 3) vehicle-frame points inside one Box's footprint
    and height range; the box is in the same frame as the points."""
    cosine = np.cos(-box.yaw)
    sine = np.sin(-box.yaw)
    relative_x = points[:, 0] - box.center_x
    relative_y = points[:, 1] - box.center_y
    local_x = cosine * relative_x - sine * relative_y
    local_y = sine * relative_x + cosine * relative_y
    inside_footprint = (np.abs(local_x) <= box.length /
                        2.0) & (np.abs(local_y) <= box.width / 2.0)
    inside_height = (np.abs(points[:, 2] - box.center_z) <= box.height / 2.0)
    return int(np.count_nonzero(inside_footprint & inside_height))


def constant_velocity_residuals(trajectory):
    """Returns (position_residuals, yaw_residuals) for one object's trajectory.

    trajectory is a list of (frame_index, center_x, center_y, yaw) tuples in
    frame order. Position residuals are metres, x then y per index; yaw
    residuals are wrapped radians. Indices next to a frame gap are skipped.
    """
    position_residuals = []
    yaw_residuals = []
    for index in range(1, len(trajectory) - 1):
        (
            previous_frame,
            previous_x,
            previous_y,
            previous_yaw,
        ) = trajectory[index - 1]
        current_frame, current_x, current_y, current_yaw = (trajectory[index])
        next_frame, next_x, next_y, next_yaw = trajectory[index + 1]
        if (current_frame - previous_frame != 1 or
                next_frame - current_frame != 1):
            continue
        # constant-velocity prediction: 2 * current - previous
        position_residuals.append(next_x - (2.0 * current_x - previous_x))
        position_residuals.append(next_y - (2.0 * current_y - previous_y))
        yaw_residuals.append(
            wrap_angle(next_yaw - (2.0 * current_yaw - previous_yaw)))
    return position_residuals, yaw_residuals


def measure_box_jitter(boxes_by_frame):
    """Returns (position_sigma in metres, yaw_sigma in radians), printing both.

    Each sigma is the standard deviation of how far a box lands from the
    constant-velocity prediction made from its two earlier frames.
    boxes_by_frame is a list with one list of GroundTruthBox per frame.
    """
    positions_by_object = {}
    for frame_index, boxes in enumerate(boxes_by_frame):
        for ground_truth_box in boxes:
            positions_by_object.setdefault(ground_truth_box.laser_object_id,
                                           []).append((
                                               frame_index,
                                               ground_truth_box.box.center_x,
                                               ground_truth_box.box.center_y,
                                               ground_truth_box.box.yaw,
                                           ))

    position_residuals = []
    yaw_residuals = []
    for trajectory in positions_by_object.values():
        trajectory_position_residuals, trajectory_yaw_residuals = (
            constant_velocity_residuals(trajectory))
        position_residuals.extend(trajectory_position_residuals)
        yaw_residuals.extend(trajectory_yaw_residuals)

    position_sigma = (float(np.std(position_residuals))
                      if position_residuals else 0.0)
    yaw_sigma = (float(np.std(yaw_residuals)) if yaw_residuals else 0.0)
    print(f"measured sigma_measurement_position: {position_sigma:.6f} m")
    print(f"measured sigma_measurement_yaw: {yaw_sigma:.6f} rad")
    return position_sigma, yaw_sigma


def write_log(path, segment_name, frames, position_sigma, yaw_sigma):
    """Writes the header and frames in the little-endian byte layout that
    read_segment_log in src/log.cpp expects; the two must change together."""
    with open(path, "wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(frames)))
        name_bytes = segment_name.encode("utf-8")
        stream.write(struct.pack("<I", len(name_bytes)))
        stream.write(name_bytes)
        stream.write(struct.pack("<2d", position_sigma, yaw_sigma))
        for frame in frames:
            stream.write(struct.pack("<q", frame["capture_time_micros"]))
            stream.write(
                struct.pack(
                    "<16d",
                    *frame["vehicle_to_world"].flatten().tolist(),
                ))
            points = frame["points"]
            stream.write(struct.pack("<I", len(points)))
            stream.write(points.astype("<f4").tobytes())
            ground_truth = frame["ground_truth"]
            stream.write(struct.pack("<I", len(ground_truth)))
            for ground_truth_box in ground_truth:
                stream.write(
                    struct.pack(
                        "<Q",
                        stable_object_id(ground_truth_box.laser_object_id)))
                stream.write(struct.pack("<B", ground_truth_box.object_class))
                stream.write(struct.pack("<7d", *ground_truth_box.box))
                stream.write(
                    struct.pack("<i", ground_truth_box.num_lidar_points_in_box))


def check_point_ratio(first_frame):
    """Prints the first frame's staged/reported point ratio, and a warning
    if it falls outside [MIN_POINT_RATIO, MAX_POINT_RATIO]."""
    staged_points_in_boxes = sum(
        count_points_in_box(first_frame["points"], ground_truth_box.box)
        for ground_truth_box in first_frame["ground_truth"])
    reported_points_in_boxes = sum(
        ground_truth_box.num_lidar_points_in_box
        for ground_truth_box in first_frame["ground_truth"])
    if reported_points_in_boxes:
        point_ratio = staged_points_in_boxes / reported_points_in_boxes
    else:
        point_ratio = 0.0
    print(f"staged/reported point ratio for first frame: {point_ratio:.4f}")
    if not MIN_POINT_RATIO <= point_ratio <= MAX_POINT_RATIO:
        print("WARNING: point ratio far from 1 - "
              "the azimuth convention is likely wrong")


def main():
    """Stages one segment: reads it, checks the first frame's point ratio,
    measures the jitter sigmas and writes the log."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required = True)
    parser.add_argument("--segment", dest = "segment_name", required = True)
    parser.add_argument("--out", required = True)
    parser.add_argument("--laser", type = int, default = 1)
    arguments = parser.parse_args()

    frames = read_components(arguments.parquet_root, arguments.segment_name,
                             arguments.laser)

    if frames:
        check_point_ratio(frames[0])

    position_sigma, yaw_sigma = measure_box_jitter(
        [frame["ground_truth"] for frame in frames])

    write_log(
        arguments.out,
        arguments.segment_name,
        frames,
        position_sigma,
        yaw_sigma,
    )


if __name__ == "__main__":
    main()
