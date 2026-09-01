#pragma once
#include <Eigen/Dense>
#include "types.hpp"

struct FilterNoise {
  // Unmodelled speed change per frame, m/s^2. Chosen.
  double sigma_acceleration = 2.0;
  // Unmodelled turn-rate change per frame, rad/s^2. Chosen.
  double sigma_yaw_acceleration = 0.5;
  // Box centre jitter, metres. Measured per segment by
  // stage_segment.py; 0 is refused.
  double sigma_measurement_position = 0.0;
  // Box heading jitter, radians. Measured per segment by
  // stage_segment.py.
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
