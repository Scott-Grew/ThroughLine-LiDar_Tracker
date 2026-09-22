// Declares the extended Kalman filter tracker.cpp runs per track:
// state initialisation, prediction, and measurement update.

#pragma once
#include <Eigen/Dense>
#include "types.hpp"

// Process and measurement noise the filter uses to weigh a
// prediction against a measurement.
struct FilterNoise {
  // Process noise: sigma of unmodelled linear (m/s^2) and yaw
  // (rad/s^2) acceleration. Chosen.
  double sigma_acceleration = 2.0;
  double sigma_yaw_acceleration = 0.5;
  // Box jitter measured per segment by stage_segment.py: position in
  // metres, yaw in radians. initial_state refuses a sigma <= 0.
  double sigma_measurement_position = 0.0;
  double sigma_measurement_yaw = 0.0;
};

// Below this yaw rate the motion model treats the path as straight,
// avoiding a divide-by-zero in the curved-path equations in filter.cpp.
constexpr double kStraightLineYawRate = 1e-4;

// Returns radians wrapped into [-pi, pi].
double wrap_angle(double radians);
// TrackState from box's first measurement; throws if a sigma is <= 0.
TrackState initial_state(const Box& box, const FilterNoise& noise);
// Advances state by dt_seconds, widening covariance by process noise.
void predict(TrackState& state, double dt_seconds, const FilterNoise& noise);
// Measurement minus predicted state, x, y, yaw wrapped to shortest turn.
Eigen::Vector3d compute_innovation(const TrackState& state,
                                   const Box& measurement);
// Expected spread of the innovation: state uncertainty plus sensor jitter.
Eigen::Matrix3d compute_innovation_covariance(const TrackState& state,
                                              const FilterNoise& noise);
// Squared Mahalanobis distance between state and measurement, unitless.
double mahalanobis_squared(const TrackState& state, const Box& measurement,
                           const FilterNoise& noise);
// Folds measurement into state via the Kalman gain, Joseph-form covariance.
void update(TrackState& state, const Box& measurement,
            const FilterNoise& noise);
