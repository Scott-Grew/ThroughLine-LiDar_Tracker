#pragma once
#include <cstdint>
#include <vector>
#include <Eigen/Dense>

// This file has no logic in it, only the shapes every other file agrees to pass around: a
// detection or ground truth box, a frame of the log, the five numbers the filter tracks for one
// object, and a track itself with its life-cycle bookkeeping. Every part of the pipeline - the
// stager, the filter, the assignment code, the tracker, the metrics and the viewer - includes this
// file and none of them define their own version of these shapes, so a field added or removed here
// is a change every one of those files has to answer for.

enum class ObjectClass : std::uint8_t { Vehicle = 1, Pedestrian = 2, Cyclist = 4 };

struct Box {
  double center_x, center_y, center_z;
  double length, width, height;
  double yaw;
};

struct Detection {
  ObjectClass object_class;
  Box box;
  float score;
};

struct GroundTruthBox {
  std::uint64_t object_id;
  ObjectClass object_class;
  Box box;
  std::int32_t lidar_points_in_box;
  std::uint8_t tracking_difficulty;
};

struct Point { float x, y, z, intensity; };

struct Frame {
  std::int64_t capture_time_micros;
  Eigen::Matrix4d vehicle_to_world;
  std::vector<Point> points;
  std::vector<GroundTruthBox> ground_truth;
  std::vector<Detection> detections;
};

struct TrackState {
  Eigen::Matrix<double, 5, 1> mean;
  Eigen::Matrix<double, 5, 5> covariance;
  double center_z, length, width, height;
};

enum class TrackStatus : std::uint8_t { Tentative, Confirmed, Coasting };

struct Track {
  std::uint64_t track_id;
  ObjectClass object_class;
  TrackStatus status;
  TrackState state;
  std::int64_t last_update_micros;
  std::int64_t state_time_micros;
  std::uint32_t hits;
  std::uint32_t consecutive_misses;
  std::vector<Eigen::Vector2d> history;
};

// Lets an ObjectClass be used as an unordered_map key, which the metrics and tracker code both
// do to keep one running total per object class. The standard library has no built-in hash for
// an enum, so without this the maps in metrics.hpp and tracker.hpp would not compile.
template <>
struct std::hash<ObjectClass> {
  std::size_t operator()(ObjectClass object_class) const noexcept {
    return std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(object_class));
  }
};
