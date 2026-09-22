// Extended Kalman filter for a track's position, yaw, speed,
// and turn rate; tracker.cpp predicts each frame, updates on match.

#include "filter.hpp"

#include <cmath>
#include <stdexcept>

namespace {

// Initial sigmas for speed (m/s) and yaw rate (rad/s), which start
// at zero unmeasured; wide so early frames correct them freely.
constexpr double kInitialSpeedSigma = 10.0;
constexpr double kInitialYawRateSigma = 1.0;

// Fraction of a newly measured box size adopted per frame; size is
// smoothed, not estimated like position and speed.
constexpr double kSizeSmoothing = 0.3;

Eigen::Matrix<double, 3, 5> build_measurement_jacobian() {
  Eigen::Matrix<double, 3, 5> jacobian = Eigen::Matrix<double, 3, 5>::Zero();
  jacobian(kPositionXIndex, kPositionXIndex) = 1.0;
  jacobian(kPositionYIndex, kPositionYIndex) = 1.0;
  jacobian(kYawIndex, kYawIndex) = 1.0;
  return jacobian;
}

Eigen::Matrix3d build_measurement_noise(const FilterNoise& noise) {
  Eigen::Matrix3d noise_covariance = Eigen::Matrix3d::Zero();
  noise_covariance(kPositionXIndex, kPositionXIndex) =
      noise.sigma_measurement_position * noise.sigma_measurement_position;
  noise_covariance(kPositionYIndex, kPositionYIndex) =
      noise.sigma_measurement_position * noise.sigma_measurement_position;
  noise_covariance(kYawIndex, kYawIndex) =
      noise.sigma_measurement_yaw * noise.sigma_measurement_yaw;
  return noise_covariance;
}

}  // namespace

double wrap_angle(double radians) {
  return std::remainder(radians, 2.0 * M_PI);
}

TrackState initial_state(const Box& box, const FilterNoise& noise) {
  if (noise.sigma_measurement_position <= 0.0 ||
      noise.sigma_measurement_yaw <= 0.0)
    throw std::invalid_argument(
        "measurement sigmas must be positive; pass --sigma-position and "
        "--sigma-yaw, or restage the segment");
  TrackState state;
  state.mean << box.center_x, box.center_y, box.yaw, 0.0, 0.0;
  state.covariance.setZero();
  state.covariance(kPositionXIndex, kPositionXIndex) =
      noise.sigma_measurement_position * noise.sigma_measurement_position;
  state.covariance(kPositionYIndex, kPositionYIndex) =
      noise.sigma_measurement_position * noise.sigma_measurement_position;
  state.covariance(kYawIndex, kYawIndex) =
      noise.sigma_measurement_yaw * noise.sigma_measurement_yaw;
  state.covariance(kSpeedIndex, kSpeedIndex) =
      kInitialSpeedSigma * kInitialSpeedSigma;
  state.covariance(kYawRateIndex, kYawRateIndex) =
      kInitialYawRateSigma * kInitialYawRateSigma;
  state.center_z = box.center_z;
  state.length = box.length;
  state.width = box.width;
  state.height = box.height;
  return state;
}

void predict(TrackState& state, double dt_seconds, const FilterNoise& noise) {
  const double yaw = state.mean(kYawIndex);
  const double speed = state.mean(kSpeedIndex);
  const double yaw_rate = state.mean(kYawRateIndex);

  Eigen::Matrix<double, 5, 5> motion_jacobian =
      Eigen::Matrix<double, 5, 5>::Identity();
  if (std::abs(yaw_rate) < kStraightLineYawRate) {
    state.mean(kPositionXIndex) += speed * std::cos(yaw) * dt_seconds;
    state.mean(kPositionYIndex) += speed * std::sin(yaw) * dt_seconds;
    motion_jacobian(kPositionXIndex, kYawIndex) =
        -speed * std::sin(yaw) * dt_seconds;
    motion_jacobian(kPositionXIndex, kSpeedIndex) = std::cos(yaw) * dt_seconds;
    motion_jacobian(kPositionYIndex, kYawIndex) =
        speed * std::cos(yaw) * dt_seconds;
    motion_jacobian(kPositionYIndex, kSpeedIndex) = std::sin(yaw) * dt_seconds;
  } else {
    // Constant turn rate path with radius r = speed / yaw_rate moves x by
    // r * (sin(yaw_next) - sin(yaw)) and y by r * (cos(yaw) - cos(yaw_next)).
    const double yaw_next = yaw + yaw_rate * dt_seconds;
    const double radius = speed / yaw_rate;
    state.mean(kPositionXIndex) +=
        radius * (std::sin(yaw_next) - std::sin(yaw));
    state.mean(kPositionYIndex) +=
        radius * (std::cos(yaw) - std::cos(yaw_next));
    // The Jacobian entries are the partial derivatives of those two moves
    // with respect to yaw, speed and yaw rate.
    motion_jacobian(kPositionXIndex, kYawIndex) =
        radius * (std::cos(yaw_next) - std::cos(yaw));
    motion_jacobian(kPositionXIndex, kSpeedIndex) =
        (std::sin(yaw_next) - std::sin(yaw)) / yaw_rate;
    motion_jacobian(kPositionXIndex, kYawRateIndex) =
        radius * dt_seconds * std::cos(yaw_next) -
        radius * (std::sin(yaw_next) - std::sin(yaw)) / yaw_rate;
    motion_jacobian(kPositionYIndex, kYawIndex) =
        radius * (std::sin(yaw_next) - std::sin(yaw));
    motion_jacobian(kPositionYIndex, kSpeedIndex) =
        (std::cos(yaw) - std::cos(yaw_next)) / yaw_rate;
    motion_jacobian(kPositionYIndex, kYawRateIndex) =
        radius * dt_seconds * std::sin(yaw_next) -
        radius * (std::cos(yaw) - std::cos(yaw_next)) / yaw_rate;
    state.mean(kYawIndex) = yaw_next;
  }
  motion_jacobian(kYawIndex, kYawRateIndex) = dt_seconds;
  state.mean(kYawIndex) = wrap_angle(state.mean(kYawIndex));

  const double half_dt_squared = 0.5 * dt_seconds * dt_seconds;

  // Process noise is Q = sa^2 g g^T + sy^2 h h^T with g = (dt^2/2 cos yaw,
  // dt^2/2 sin yaw, 0, dt, 0) and h = (0, 0, dt^2/2, 0, dt).
  Eigen::Matrix<double, 5, 1> acceleration_direction =
      Eigen::Matrix<double, 5, 1>::Zero();
  acceleration_direction(kPositionXIndex) = half_dt_squared * std::cos(yaw);
  acceleration_direction(kPositionYIndex) = half_dt_squared * std::sin(yaw);
  acceleration_direction(kSpeedIndex) = dt_seconds;

  Eigen::Matrix<double, 5, 1> yaw_acceleration_direction =
      Eigen::Matrix<double, 5, 1>::Zero();
  yaw_acceleration_direction(kYawIndex) = half_dt_squared;
  yaw_acceleration_direction(kYawRateIndex) = dt_seconds;

  const Eigen::Matrix<double, 5, 5> process_noise =
      noise.sigma_acceleration * noise.sigma_acceleration *
          acceleration_direction * acceleration_direction.transpose() +
      noise.sigma_yaw_acceleration * noise.sigma_yaw_acceleration *
          yaw_acceleration_direction * yaw_acceleration_direction.transpose();

  // P = F P F^T + Q for motion Jacobian F.
  state.covariance =
      motion_jacobian * state.covariance * motion_jacobian.transpose() +
      process_noise;
}

Eigen::Vector3d compute_innovation(const TrackState& state,
                                   const Box& measurement) {
  Eigen::Vector3d innovation;
  innovation(kPositionXIndex) =
      measurement.center_x - state.mean(kPositionXIndex);
  innovation(kPositionYIndex) =
      measurement.center_y - state.mean(kPositionYIndex);
  innovation(kYawIndex) = wrap_angle(measurement.yaw - state.mean(kYawIndex));
  return innovation;
}

Eigen::Matrix3d compute_innovation_covariance(const TrackState& state,
                                              const FilterNoise& noise) {
  const Eigen::Matrix<double, 3, 5> measurement_jacobian =
      build_measurement_jacobian();
  // S = H P H^T + R for measurement Jacobian H and sensor noise R.
  return measurement_jacobian * state.covariance *
             measurement_jacobian.transpose() +
         build_measurement_noise(noise);
}

double mahalanobis_squared(const TrackState& state, const Box& measurement,
                           const FilterNoise& noise) {
  const Eigen::Vector3d innovation = compute_innovation(state, measurement);
  const Eigen::Matrix3d innovation_covariance =
      compute_innovation_covariance(state, noise);
  // d^2 = y^T S^-1 y for innovation y, solved rather than inverted.
  return innovation.transpose() *
         innovation_covariance.ldlt().solve(innovation);
}

void update(TrackState& state, const Box& measurement,
            const FilterNoise& noise) {
  const Eigen::Matrix<double, 3, 5> measurement_jacobian =
      build_measurement_jacobian();
  const Eigen::Vector3d innovation = compute_innovation(state, measurement);
  const Eigen::Matrix3d innovation_covariance =
      compute_innovation_covariance(state, noise);
  // K = P H^T S^-1.
  const Eigen::Matrix<double, 5, 3> kalman_gain =
      state.covariance * measurement_jacobian.transpose() *
      innovation_covariance.inverse();

  state.mean += kalman_gain * innovation;
  state.mean(kYawIndex) = wrap_angle(state.mean(kYawIndex));

  // Joseph form P = (I - K H) P (I - K H)^T + K R K^T keeps the covariance
  // symmetric and positive definite.
  const Eigen::Matrix<double, 5, 5> joseph_factor =
      Eigen::Matrix<double, 5, 5>::Identity() -
      kalman_gain * measurement_jacobian;
  state.covariance =
      joseph_factor * state.covariance * joseph_factor.transpose() +
      kalman_gain * build_measurement_noise(noise) * kalman_gain.transpose();

  state.center_z += kSizeSmoothing * (measurement.center_z - state.center_z);
  state.length += kSizeSmoothing * (measurement.length - state.length);
  state.width += kSizeSmoothing * (measurement.width - state.width);
  state.height += kSizeSmoothing * (measurement.height - state.height);
}
