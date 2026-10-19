// Shared data shapes for detections, frames, filter state and tracks. Each
// module includes it.

#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include <Eigen/Dense>

// Microseconds in one second, shared by all timestamp conversions.
constexpr std::int64_t kMicrosPerSecond = 1'000'000;

// The object classes a detection or track can belong to; the
// values are Waymo's own type codes, stored as-is in the log.
enum class ObjectClass : std::uint8_t {
  Vehicle = 1,
  Pedestrian = 2,
  Cyclist = 4,
};

// A 3D oriented box with center, extents and yaw. Detections hold it in the
// vehicle frame, tracks in the world frame.
struct Box {
  double center_x = 0.0;
  double center_y = 0.0;
  double center_z = 0.0;
  double length = 0.0;
  double width = 0.0;
  double height = 0.0;
  double yaw = 0.0;
};

// One detector output in a frame.
struct Detection {
  ObjectClass object_class = ObjectClass::Vehicle;
  Box box;
};

// One lidar point in the vehicle frame.
struct Point {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
};

// One frame of the log with its capture time, the vehicle's pose in the
// world frame, its lidar points and the detections made in it.
struct Frame {
  std::int64_t capture_time_micros = 0;
  Eigen::Matrix4d vehicle_to_world = Eigen::Matrix4d::Identity();
  std::vector<Point> points;
  std::vector<Detection> detections;
};

// Positions of x, y, yaw, speed and yaw rate in the filter state and its
// covariance. A measured box fills the first three.
constexpr int kStateX = 0;
constexpr int kStateY = 1;
constexpr int kStateYaw = 2;
constexpr int kStateSpeed = 3;
constexpr int kStateYawRate = 4;
constexpr int kStateSize = 5;
constexpr int kMeasuredSize = 3;

using StateVector = Eigen::Matrix<double, kStateSize, 1>;
using StateMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;

// Filter mean and covariance over x, y, yaw, speed and yaw rate in the
// world frame, in m, rad, m/s and rad/s, the time they hold for, the
// object class's acceleration sigma, and the smoothed box size.
struct TrackState {
  StateVector mean = StateVector::Zero();
  StateMatrix covariance = StateMatrix::Zero();
  std::int64_t time_micros = 0;
  double acceleration_sigma = 0.0;
  double center_z = 0.0;
  double length = 0.0;
  double width = 0.0;
  double height = 0.0;

  Eigen::Vector2d position() const { return mean.head<2>(); }
  double yaw() const { return mean(kStateYaw); }
};

// A track is tentative, confirmed, coasting on a miss, or dead and
// about to be erased.
enum class TrackStatus : std::uint8_t { Tentative, Confirmed, Coasting, Dead };

// One tracked object with its identity, class, life-cycle state, current
// filter state and recent position history for drawing.
struct Track {
  std::uint64_t track_id = 0;
  ObjectClass object_class = ObjectClass::Vehicle;
  TrackStatus status = TrackStatus::Tentative;
  TrackState state;
  std::uint32_t hits = 0;
  std::uint32_t consecutive_misses = 0;
  std::vector<Eigen::Vector2d> history;
};

// The vehicle's own yaw is atan2(R10, R00) of the pose's rotation.
inline double vehicle_yaw(const Eigen::Matrix4d& vehicle_to_world) {
  return std::atan2(vehicle_to_world(1, 0), vehicle_to_world(0, 0));
}
