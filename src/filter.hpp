#pragma once
#include <Eigen/Dense>
#include "types.hpp"

// Declares the extended Kalman filter tracker.cpp runs per track:
// state initialisation, prediction, and measurement update.

// Process and measurement noise the filter uses to weigh a
// prediction against a measurement.
struct FilterNoise {
  // Unmodelled acceleration per frame; grows uncertainty between
  // measurements. Chosen: linear in m/s^2, yaw in rad/s^2.
  double sigma_acceleration = 2.0;
  double sigma_yaw_acceleration = 0.5;
  // Box jitter, measured per segment by stage_segment.py. Position in
  // metres, yaw in radians; a zero position sigma is refused.
  double sigma_measurement_position = 0.0;
  double sigma_measurement_yaw = 0.0;
};

double wrap_angle(double radians);
TrackState initial_state(const Box& box, const FilterNoise& noise);
void predict(TrackState& state, double dt_seconds,
             const FilterNoise& noise);
Eigen::Vector3d innovation(const TrackState& state,
                           const Box& measurement);
Eigen::Matrix3d innovation_covariance(const TrackState& state,
                                      const FilterNoise& noise);
double mahalanobis_squared(const TrackState& state,
                           const Box& measurement,
                           const FilterNoise& noise);
void update(TrackState& state, const Box& measurement,
            const FilterNoise& noise);
