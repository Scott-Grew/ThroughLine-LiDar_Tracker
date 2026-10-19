// Declares the extended Kalman filter each track runs over x, y, yaw,
// speed and yaw rate. tracker.cpp and recording.cpp hold one each.

#pragma once
#include <cstdint>
#include <Eigen/Dense>
#include "types.hpp"

// Process and measurement noise the filter uses to weigh a
// prediction against a measurement.
struct FilterNoise {
  // Process noise is the sigma of unmodelled linear acceleration in
  // m/s^2 and yaw acceleration in rad/s^2. Chosen.
  double sigma_acceleration = 2.0;
  double sigma_yaw_acceleration = 0.5;
  // Measurement sigmas are the box jitter stage_segment.py measured for
  // the segment, in metres and radians. MotionFilter refuses <= 0.
  double sigma_measurement_position = 0.0;
  double sigma_measurement_yaw = 0.0;
};

// Below this yaw rate the motion model treats the path as straight,
// avoiding a divide-by-zero in the curved-path equations in filter.cpp.
constexpr double kStraightLineYawRate = 1e-4;

double wrap_angle(double radians);

// Constant-turn-rate extended Kalman filter. It holds the noise, so each
// call takes only the state it works on.
class MotionFilter {
 public:
  explicit MotionFilter(FilterNoise noise);

  // A state at the measured box, with speed and yaw rate unknown.
  [[nodiscard]] TrackState start_state(Box box, std::int64_t time_micros) const;

  // Moves the state forward to time_micros; a state already at or past
  // that time is left alone.
  void advance(TrackState& state, std::int64_t time_micros) const;
  void predict(TrackState& state, double dt_seconds) const;

  // Squared Mahalanobis distance between the state and a measured box.
  [[nodiscard]] double distance_squared(const TrackState& state, Box box) const;
  void update(TrackState& state, Box box) const;

 private:
  FilterNoise noise_;
  Eigen::Matrix3d measurement_noise_;
};
