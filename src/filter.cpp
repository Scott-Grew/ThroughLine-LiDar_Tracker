#include "filter.hpp"

double wrap_angle(double radians) {
  (void)radians;
  return 0.0;
}

TrackState initial_state(const Box& box, const FilterNoise& noise) {
  (void)box;
  (void)noise;
  return TrackState{};
}

void predict(TrackState& state, double dt_seconds, const FilterNoise& noise) {
  (void)state;
  (void)dt_seconds;
  (void)noise;
}

Eigen::Vector3d innovation(const TrackState& state, const Box& measurement) {
  (void)state;
  (void)measurement;
  return Eigen::Vector3d::Zero();
}

Eigen::Matrix3d innovation_covariance(const TrackState& state, const FilterNoise& noise) {
  (void)state;
  (void)noise;
  return Eigen::Matrix3d::Zero();
}

double mahalanobis_squared(const TrackState& state, const Box& measurement, const FilterNoise& noise) {
  (void)state;
  (void)measurement;
  (void)noise;
  return 0.0;
}

void update(TrackState& state, const Box& measurement, const FilterNoise& noise) {
  (void)state;
  (void)measurement;
  (void)noise;
}
