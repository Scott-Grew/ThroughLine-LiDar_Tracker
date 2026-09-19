#include "filter.hpp"

#include <cmath>
#include <stdexcept>

// Extended Kalman filter for a track's position, heading, speed,
// and turn rate; tracker.cpp predicts every frame, updates on match.

namespace {

// Initial uncertainty for speed and yaw rate, both unmeasured and
// guessed as zero; wide enough to let early frames correct freely.
constexpr double kInitialSpeedSigma = 10.0;
constexpr double kInitialYawRateSigma = 1.0;

// Below this yaw rate the motion model treats the path as straight,
// avoiding a divide-by-zero in the curved-path equations below.
constexpr double kStraightLineYawRate = 1e-4;

// Fraction of a newly measured box size adopted per frame; size is
// smoothed, not estimated like position and speed.
constexpr double kSizeSmoothing = 0.3;

// Maps the five-number state to the three the sensor measures:
// position and heading, never speed or turn rate.
Eigen::Matrix<double, 3, 5> measurement_jacobian() {
  Eigen::Matrix<double, 3, 5> matrix =
      Eigen::Matrix<double, 3, 5>::Zero();
  matrix(0, 0) = 1.0;
  matrix(1, 1) = 1.0;
  matrix(2, 2) = 1.0;
  return matrix;
}

// Measurement noise covariance from the sigmas the staging script
// measured for this segment.
Eigen::Matrix3d measurement_noise(const FilterNoise& noise) {
  Eigen::Matrix3d matrix = Eigen::Matrix3d::Zero();
  matrix(0, 0) = noise.sigma_measurement_position *
                 noise.sigma_measurement_position;
  matrix(1, 1) = noise.sigma_measurement_position *
                 noise.sigma_measurement_position;
  matrix(2, 2) =
      noise.sigma_measurement_yaw * noise.sigma_measurement_yaw;
  return matrix;
}

}  // namespace

// Wraps an angle into (-pi, pi]. Every heading difference in this
// file must be wrapped, or a near-identical heading reads as huge.
double wrap_angle(double radians) {
  return std::remainder(radians, 2.0 * M_PI);
}

// Builds a track's starting state from its first box, with speed and
// yaw rate at zero. Throws if either measurement sigma is <= 0.
TrackState initial_state(const Box& box, const FilterNoise& noise) {
  if (noise.sigma_measurement_position <= 0.0 ||
      noise.sigma_measurement_yaw <= 0.0)
    throw std::invalid_argument("measurement noise must be positive");
  TrackState state;
  state.mean << box.center_x, box.center_y, box.yaw, 0.0, 0.0;
  state.covariance.setZero();
  state.covariance(0, 0) = noise.sigma_measurement_position *
                           noise.sigma_measurement_position;
  state.covariance(1, 1) = noise.sigma_measurement_position *
                           noise.sigma_measurement_position;
  state.covariance(2, 2) =
      noise.sigma_measurement_yaw * noise.sigma_measurement_yaw;
  state.covariance(3, 3) = kInitialSpeedSigma * kInitialSpeedSigma;
  state.covariance(4, 4) =
      kInitialYawRateSigma * kInitialYawRateSigma;
  state.center_z = box.center_z;
  state.length = box.length;
  state.width = box.width;
  state.height = box.height;
  return state;
}

// Advances the state by dt_seconds without a measurement, widening
// covariance by process noise coupled through the current heading.
void predict(TrackState& state, double dt_seconds,
             const FilterNoise& noise) {
  const double yaw = state.mean(2);
  const double speed = state.mean(3);
  const double yaw_rate = state.mean(4);

  Eigen::Matrix<double, 5, 5> jacobian =
      Eigen::Matrix<double, 5, 5>::Identity();
  if (std::abs(yaw_rate) < kStraightLineYawRate) {
    state.mean(0) += speed * std::cos(yaw) * dt_seconds;
    state.mean(1) += speed * std::sin(yaw) * dt_seconds;
    jacobian(0, 2) = -speed * std::sin(yaw) * dt_seconds;
    jacobian(0, 3) = std::cos(yaw) * dt_seconds;
    jacobian(1, 2) = speed * std::cos(yaw) * dt_seconds;
    jacobian(1, 3) = std::sin(yaw) * dt_seconds;
  } else {
    const double yaw_next = yaw + yaw_rate * dt_seconds;
    const double radius = speed / yaw_rate;
    state.mean(0) += radius * (std::sin(yaw_next) - std::sin(yaw));
    state.mean(1) += radius * (std::cos(yaw) - std::cos(yaw_next));
    jacobian(0, 2) = radius * (std::cos(yaw_next) - std::cos(yaw));
    jacobian(0, 3) = (std::sin(yaw_next) - std::sin(yaw)) / yaw_rate;
    jacobian(0, 4) =
        radius * dt_seconds * std::cos(yaw_next) -
        radius * (std::sin(yaw_next) - std::sin(yaw)) / yaw_rate;
    jacobian(1, 2) = radius * (std::sin(yaw_next) - std::sin(yaw));
    jacobian(1, 3) = (std::cos(yaw) - std::cos(yaw_next)) / yaw_rate;
    jacobian(1, 4) =
        radius * dt_seconds * std::sin(yaw_next) -
        radius * (std::cos(yaw) - std::cos(yaw_next)) / yaw_rate;
    state.mean(2) = yaw_next;
  }
  jacobian(2, 4) = dt_seconds;
  state.mean(2) = wrap_angle(state.mean(2));

  Eigen::Matrix<double, 5, 1> acceleration_direction;
  acceleration_direction
      << 0.5 * dt_seconds * dt_seconds * std::cos(yaw),
      0.5 * dt_seconds * dt_seconds * std::sin(yaw), 0.0, dt_seconds,
      0.0;
  Eigen::Matrix<double, 5, 1> yaw_acceleration_direction;
  yaw_acceleration_direction << 0.0, 0.0,
      0.5 * dt_seconds * dt_seconds, 0.0, dt_seconds;
  const Eigen::Matrix<double, 5, 5> process_noise =
      noise.sigma_acceleration * noise.sigma_acceleration *
          acceleration_direction *
          acceleration_direction.transpose() +
      noise.sigma_yaw_acceleration * noise.sigma_yaw_acceleration *
          yaw_acceleration_direction *
          yaw_acceleration_direction.transpose();

  state.covariance =
      jacobian * state.covariance * jacobian.transpose() +
      process_noise;
}

// Difference between a measured box and the predicted state, with
// the heading difference wrapped to its shortest direction.
Eigen::Vector3d innovation(const TrackState& state,
                           const Box& measurement) {
  Eigen::Vector3d difference;
  difference(0) = measurement.center_x - state.mean(0);
  difference(1) = measurement.center_y - state.mean(1);
  difference(2) = wrap_angle(measurement.yaw - state.mean(2));
  return difference;
}

// Expected spread of the innovation, combining the state's own
// uncertainty with how much the sensor itself jitters.
Eigen::Matrix3d innovation_covariance(const TrackState& state,
                                      const FilterNoise& noise) {
  const Eigen::Matrix<double, 3, 5> observation =
      measurement_jacobian();
  return observation * state.covariance * observation.transpose() +
         measurement_noise(noise);
}

// Innovation expressed in units of standard deviations rather than
// metres; the tracker gates and rejects matches on this distance.
double mahalanobis_squared(const TrackState& state,
                           const Box& measurement,
                           const FilterNoise& noise) {
  const Eigen::Vector3d difference = innovation(state, measurement);
  const Eigen::Matrix3d covariance =
      innovation_covariance(state, noise);
  return difference.transpose() * covariance.ldlt().solve(difference);
}

// Folds a matched box into the state via the Kalman gain, using the
// numerically stable Joseph form for the covariance update.
void update(TrackState& state, const Box& measurement,
            const FilterNoise& noise) {
  const Eigen::Matrix<double, 3, 5> observation =
      measurement_jacobian();
  const Eigen::Vector3d difference = innovation(state, measurement);
  const Eigen::Matrix3d covariance =
      innovation_covariance(state, noise);
  const Eigen::Matrix<double, 5, 3> gain = state.covariance *
                                           observation.transpose() *
                                           covariance.inverse();

  state.mean += gain * difference;
  state.mean(2) = wrap_angle(state.mean(2));

  const Eigen::Matrix<double, 5, 5> factor =
      Eigen::Matrix<double, 5, 5>::Identity() - gain * observation;
  state.covariance =
      factor * state.covariance * factor.transpose() +
      gain * measurement_noise(noise) * gain.transpose();

  state.center_z +=
      kSizeSmoothing * (measurement.center_z - state.center_z);
  state.length +=
      kSizeSmoothing * (measurement.length - state.length);
  state.width += kSizeSmoothing * (measurement.width - state.width);
  state.height +=
      kSizeSmoothing * (measurement.height - state.height);
}
