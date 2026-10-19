"""Stages one Waymo v2 segment as an MCAP log for the C++ tracker, with lidar
points from the parquet tables and boxes from CasA's published detections."""

import argparse
import pickle

import numpy as np
import pyarrow.parquet as pq
import tracker_pb2
from mcap_protobuf.writer import Writer
from waymo_boxes import TRACKED_OBJECT_CLASSES, read_ground_truth_boxes

# CasA's class names mapped to Waymo's type codes, and the score below
# which its detections are dropped. Chosen.
CLASS_CODES = {"Vehicle": 1, "Pedestrian": 2, "Cyclist": 4}
MIN_SCORE = 0.5

# The only classes a CasA result pickle may build, so loading it runs no code.
NUMPY_CLASSES = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}

# Scale from the median absolute deviation to a Gaussian sigma.
MAD_TO_SIGMA = 1.4826

# Accepted range for staged points in boxes over Waymo's reported count.
MIN_POINT_RATIO = 0.5
MAX_POINT_RATIO = 1.5

RANGE_IMAGE = "[LiDARComponent].range_image_return1"
CALIBRATION = "[LiDARCalibrationComponent]"


class NumpyOnlyUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if (module, name) not in NUMPY_CLASSES:
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        return super().find_class(module, name)


def wrap_angle(radians):
    return np.angle(np.exp(1j * radians))


def read_table(parquet_root, component, segment_name, **options):
    path = f"{parquet_root}/{component}/{segment_name}.parquet"
    return pq.read_table(path, **options)


def read_calibration(parquet_root, segment_name, laser):
    """Returns the laser's sensor-to-vehicle transform and its beam
    inclinations as (values or None, min, max)."""
    table = read_table(parquet_root, "lidar_calibration", segment_name,
                       filters = [("key.laser_name", "=", laser)])
    row = table.to_pylist()[0]
    extrinsic = np.array(row[f"{CALIBRATION}.extrinsic.transform"])
    values = row[f"{CALIBRATION}.beam_inclination.values"]
    low = row[f"{CALIBRATION}.beam_inclination.min"]
    high = row[f"{CALIBRATION}.beam_inclination.max"]
    return extrinsic.reshape(4, 4), values, low, high


def range_image_to_points(range_image, calibration):
    """Converts one (rows, columns, channels) range image to (N, 3) float32
    vehicle-frame points; channel 0 is range, and -1 marks no return."""
    extrinsic, values, low, high = calibration
    height, width, _ = range_image.shape
    if values and len(values) == height:
        inclinations = np.array(values[::-1])
    else:
        inclinations = np.linspace(high, low, height)
    # Column c looks along azimuth pi - (c + 0.5) 2 pi / width in the sensor
    # frame, less the extrinsic's own yaw.
    sensor_yaw = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    columns = np.arange(width)
    azimuths = np.pi - (columns + 0.5) * 2.0 * np.pi / width - sensor_yaw
    inclination, azimuth = np.meshgrid(inclinations, azimuths, indexing = "ij")

    ranges = range_image[:, :, 0]
    has_return = ranges > 0.0
    distance = ranges[has_return]
    elevation = inclination[has_return]
    heading = azimuth[has_return]
    # x = r cos(incl) cos(az), y = r cos(incl) sin(az), z = r sin(incl).
    sensor_points = np.stack([
        distance * np.cos(elevation) * np.cos(heading),
        distance * np.cos(elevation) * np.sin(heading),
        distance * np.sin(elevation),
        np.ones_like(distance),
    ])
    vehicle_points = extrinsic @ sensor_points
    return vehicle_points[:3].T.astype(np.float32)


def read_points(parquet_root, segment_name, laser):
    """Returns {capture time in microseconds: (N, 3) vehicle-frame points}."""
    calibration = read_calibration(parquet_root, segment_name, laser)
    columns = [
        "key.frame_timestamp_micros",
        f"{RANGE_IMAGE}.values",
        f"{RANGE_IMAGE}.shape",
    ]
    table = read_table(parquet_root, "lidar", segment_name, columns = columns,
                       filters = [("key.laser_name", "=", laser)])
    times = table.column(columns[0]).to_pylist()
    points_by_time = {}
    for row, capture_time in enumerate(times):
        flat_values = table.column(columns[1])[row].values.to_numpy()
        shape = table.column(columns[2])[row].as_py()
        range_image = flat_values.reshape(shape)
        points = range_image_to_points(range_image, calibration)
        points_by_time[capture_time] = points
    return points_by_time


def read_poses(parquet_root, segment_name):
    """Returns {capture time in microseconds: 16-element row-major pose}."""
    columns = [
        "key.frame_timestamp_micros",
        "[VehiclePoseComponent].world_from_vehicle.transform",
    ]
    table = read_table(parquet_root, "vehicle_pose", segment_name,
                       columns = columns)
    times = table.column(columns[0]).to_pylist()
    poses = table.column(columns[1]).to_pylist()
    return dict(zip(times, poses, strict = True))


def read_detections(result_path, segment_name):
    """Returns CasA's detections scoring at least MIN_SCORE, in frame order,
    one (classes, (N, 7) boxes) pair per frame, boxes in the vehicle frame."""
    with open(result_path, "rb") as stream:
        results = NumpyOnlyUnpickler(stream).load()
    sequence = f"segment-{segment_name}_with_camera_labels"
    segment_results = [r for r in results if str(r["seq_id"]) == sequence]
    segment_results.sort(key = lambda result: int(result["frame_id"]))
    detections = []
    for result in segment_results:
        kept = result["score"] >= MIN_SCORE
        names = result["name"][kept]
        classes = np.array([CLASS_CODES[name] for name in names])
        boxes = result["boxes_lidar"][kept].astype(np.float64)
        detections.append((classes, boxes))
    return detections


def scored_labels(ground_truth):
    """Returns {capture time in microseconds: (classes, (N, 7) boxes)} for the
    tracked labels with lidar support, the boxes the evaluator scores."""
    rows_by_time = {}
    for label in ground_truth:
        if label.object_class not in TRACKED_OBJECT_CLASSES:
            continue
        if label.num_lidar_points_in_box <= 0:
            continue
        rows = rows_by_time.setdefault(label.capture_time_micros, [])
        rows.append((label.object_class, *label.box))
    labels_by_time = {}
    for capture_time, rows in rows_by_time.items():
        table = np.array(rows)
        labels_by_time[capture_time] = (table[:, 0], table[:, 1:])
    return labels_by_time


def matched_errors(detections, labels):
    """Returns (position errors, yaw errors) of the detections that are the
    mutual nearest same-class neighbour of a label in one frame."""
    detection_classes, detection_boxes = detections
    label_classes, label_boxes = labels
    if len(detection_boxes) == 0:
        return np.empty(0), np.empty(0)
    offsets = detection_boxes[:, None, :2] - label_boxes[None, :, :2]
    distances = np.linalg.norm(offsets, axis = 2)
    same_class = detection_classes[:, None] == label_classes[None, :]
    distances[~same_class] = np.inf
    nearest_label = np.argmin(distances, axis = 1)
    nearest_detection = np.argmin(distances, axis = 0)
    detection_indices = np.arange(len(detection_boxes))
    mutual = nearest_detection[nearest_label] == detection_indices
    mutual &= np.isfinite(distances[detection_indices, nearest_label])
    matched_labels = label_boxes[nearest_label[mutual]]
    errors = detection_boxes[mutual] - matched_labels
    yaw_errors = wrap_angle(errors[:, 6])
    return errors[:, :2].ravel(), yaw_errors


def robust_sigma(errors):
    deviation = np.abs(errors - np.median(errors))
    return float(MAD_TO_SIGMA * np.median(deviation))


def count_points_in_box(points, box):
    """Counts the vehicle-frame points inside one box."""
    # Rotating by -yaw about the centre turns the footprint test into an
    # axis-aligned bound.
    relative_x = points[:, 0] - box.center_x
    relative_y = points[:, 1] - box.center_y
    cosine = np.cos(box.yaw)
    sine = np.sin(box.yaw)
    local_x = cosine * relative_x + sine * relative_y
    local_y = cosine * relative_y - sine * relative_x
    inside = np.abs(local_x) <= box.length / 2
    inside &= np.abs(local_y) <= box.width / 2
    inside &= np.abs(points[:, 2] - box.center_z) <= box.height / 2
    return int(np.count_nonzero(inside))


def check_point_ratio(ground_truth, first_time, points):
    """Warns if the first frame's staged points in the labelled boxes stray
    from Waymo's own count, the sign of a wrong azimuth convention."""
    staged = 0
    reported = 0
    for label in ground_truth:
        if label.capture_time_micros != first_time:
            continue
        if label.object_class not in TRACKED_OBJECT_CLASSES:
            continue
        staged += count_points_in_box(points, label.box)
        reported += label.num_lidar_points_in_box
    ratio = staged / reported if reported else 0.0
    print(f"staged/reported point ratio for first frame: {ratio:.4f}")
    if not MIN_POINT_RATIO <= ratio <= MAX_POINT_RATIO:
        print("WARNING: point ratio far from 1, check the azimuth convention")


def detector_sigmas(times, detections, labels_by_time):
    """Returns the detector's robust position and yaw sigma over the segment,
    from detections matched to labels."""
    position_errors = []
    yaw_errors = []
    for capture_time, frame_detections in zip(times, detections, strict = True):
        if capture_time not in labels_by_time:
            continue
        labels = labels_by_time[capture_time]
        frame_position, frame_yaw = matched_errors(frame_detections, labels)
        position_errors.append(frame_position)
        yaw_errors.append(frame_yaw)
    position_sigma = robust_sigma(np.concatenate(position_errors))
    yaw_sigma = robust_sigma(np.concatenate(yaw_errors))
    return position_sigma, yaw_sigma


def frame_message(capture_time, pose, points, detections):
    message = tracker_pb2.Frame(capture_time_micros = capture_time)
    message.vehicle_to_world.extend(pose)
    message.points.extend(points.ravel())
    classes, boxes = detections
    for object_class, box in zip(classes, boxes, strict = True):
        detection = message.detections.add(object_class = int(object_class))
        target = detection.box
        (target.center_x, target.center_y, target.center_z, target.length,
         target.width, target.height, target.yaw) = box
    return message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required = True)
    parser.add_argument("--segment", dest = "segment_name", required = True)
    parser.add_argument("--detections", required = True)
    parser.add_argument("--out", required = True)
    parser.add_argument("--laser", type = int, default = 1)
    arguments = parser.parse_args()
    parquet_root = arguments.parquet_root
    segment_name = arguments.segment_name

    points_by_time = read_points(parquet_root, segment_name, arguments.laser)
    poses = read_poses(parquet_root, segment_name)
    times = sorted(time for time in points_by_time if time in poses)
    detections = read_detections(arguments.detections, segment_name)
    if len(detections) != len(times):
        raise SystemExit(f"{len(detections)} detection frames for "
                         f"{len(times)} lidar frames")

    ground_truth = read_ground_truth_boxes(parquet_root, segment_name)
    check_point_ratio(ground_truth, times[0], points_by_time[times[0]])
    labels_by_time = scored_labels(ground_truth)
    position_sigma, yaw_sigma = detector_sigmas(times, detections,
                                                labels_by_time)
    print(f"detector sigma: {position_sigma:.4f} m, {yaw_sigma:.4f} rad")

    segment = tracker_pb2.Segment(name = segment_name,
                                  position_sigma = position_sigma,
                                  yaw_sigma = yaw_sigma)
    with Writer(arguments.out) as writer:
        first_nanoseconds = times[0] * 1000
        writer.write_message("/segment", segment, log_time = first_nanoseconds,
                             publish_time = first_nanoseconds)
        for capture_time, frame_detections in zip(times, detections,
                                                  strict = True):
            pose = poses[capture_time]
            points = points_by_time[capture_time]
            message = frame_message(capture_time, pose, points,
                                    frame_detections)
            nanoseconds = capture_time * 1000
            writer.write_message("/frames", message, log_time = nanoseconds,
                                 publish_time = nanoseconds)


if __name__ == "__main__":
    main()
