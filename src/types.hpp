#pragma once
#include <cstdint>
#include <vector>
#include <Eigen/Dense>

// Shared data shapes used across the pipeline: detections, ground
// truth, frames, filter state and tracks. Every module includes it.

// The object classes a detection or track can belong to; the
// values are Waymo's own type codes, stored as-is in the log.
enum class ObjectClass : std::uint8_t {
  Vehicle = 1,
  Pedestrian = 2,
  Cyclist = 4
};

// A 3D oriented box: center, extents and yaw. The frame it is
// expressed in depends on which struct holds it.
struct Box {
  double center_x, center_y, center_z;
  double length, width, height;
  double yaw;
};

// One perceived object in a frame, with the detector's confidence.
struct Detection {
  ObjectClass object_class;
  Box box;
  float score;
};

// One labeled ground-truth object and its lidar point support.
struct GroundTruthBox {
  std::uint64_t object_id;
  ObjectClass object_class;
  Box box;
  std::int32_t lidar_points_in_box;
};

// One lidar point in the sensor's local (vehicle) frame.
struct Point {
  float x, y, z;
};

// One frame of the log: capture time, the vehicle's pose in world,
// its lidar points and the ground-truth boxes visible in it.
struct Frame {
  std::int64_t capture_time_micros;
  Eigen::Matrix4d vehicle_to_world;
  std::vector<Point> points;
  std::vector<GroundTruthBox> ground_truth;
};

// Filter mean and covariance over x, y, yaw, speed and yaw rate in
// the world frame (m, rad, m/s, rad/s), plus smoothed box size.
struct TrackState {
  Eigen::Matrix<double, 5, 1> mean;
  Eigen::Matrix<double, 5, 5> covariance;
  double center_z, length, width, height;
};

// A track's life-cycle stage: tentative, confirmed or coasting.
enum class TrackStatus : std::uint8_t {
  Tentative,
  Confirmed,
  Coasting
};

// One tracked object: identity, class, life-cycle state, current
// filter state and its recent position history for drawing.
struct Track {
  std::uint64_t track_id;
  ObjectClass object_class;
  TrackStatus status;
  TrackState state;
  std::int64_t state_time_micros;
  std::uint32_t hits;
  std::uint32_t consecutive_misses;
  std::vector<Eigen::Vector2d> history;
};
