// Declares the extended Kalman filter that tracker.cpp runs per track
// for state initialisation, prediction and measurement update.

#pragma once
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
  // the segment, in metres and radians. initial_state refuses <= 0.
  double sigma_measurement_position = 0.0;
  double sigma_measurement_yaw = 0.0;
};

// Below this yaw rate the motion model treats the path as straight,
// avoiding a divide-by-zero in the curved-path equations in filter.cpp.
constexpr double kStraightLineYawRate = 1e-4;

double wrap_angle(double radians);
TrackState initial_state(const Box& box, const FilterNoise& noise);
void predict(TrackState& state, double dt_seconds, const FilterNoise& noise);
Eigen::Vector3d compute_innovation(const TrackState& state,
                                   const Box& measurement);
Eigen::Matrix3d compute_innovation_covariance(const TrackState& state,
                                              const FilterNoise& noise);
double mahalanobis_squared(const TrackState& state, const Box& measurement,
                           const FilterNoise& noise);
void update(TrackState& state, const Box& measurement,
            const FilterNoise& noise);
