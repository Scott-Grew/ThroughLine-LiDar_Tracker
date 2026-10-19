// Extended Kalman filter for a track's position, yaw, speed and turn rate;
// tracker.cpp predicts every track each frame and updates the matched ones.

#include "filter.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <Eigen/Dense>
#include "types.hpp"

namespace {

// Initial sigmas for speed (m/s) and yaw rate (rad/s), which start
// at zero unmeasured; wide so early frames correct them freely.
constexpr double kInitialSpeedSigma = 10.0;
constexpr double kInitialYawRateSigma = 1.0;

// Fraction of a newly measured box size adopted per frame; size is
// smoothed, not estimated like position and speed.
constexpr double kSizeSmoothing = 0.3;

Eigen::Vector3d innovation(const TrackState& state, const Box& box) {
  const double x_error = box.center_x - state.mean(kStateX);
  const double y_error = box.center_y - state.mean(kStateY);
  const double yaw_error = wrap_angle(box.yaw - state.yaw());
  return {x_error, y_error, yaw_error};
}

}  // namespace

double wrap_angle(double radians) {
  return std::remainder(radians, 2.0 * std::numbers::pi);
}

MotionFilter::MotionFilter(FilterNoise noise) : noise_(noise) {
  const double position_sigma = noise.sigma_measurement_position;
  const double yaw_sigma = noise.sigma_measurement_yaw;
  if (position_sigma <= 0.0 || yaw_sigma <= 0.0)
    throw std::invalid_argument(
        "measurement sigmas must be positive; pass --sigma-position and "
        "--sigma-yaw, or restage the segment");
  const double position_variance = position_sigma * position_sigma;
  const double yaw_variance = yaw_sigma * yaw_sigma;
  const Eigen::Vector3d variances(position_variance, position_variance,
                                  yaw_variance);
  measurement_noise_ = variances.asDiagonal();
}

TrackState MotionFilter::start_state(Box box, std::int64_t time_micros) const {
  StateVector variances;
  variances << measurement_noise_.diagonal(),
      kInitialSpeedSigma * kInitialSpeedSigma,
      kInitialYawRateSigma * kInitialYawRateSigma;

  TrackState state;
  state.mean << box.center_x, box.center_y, box.yaw, 0.0, 0.0;
  state.covariance = variances.asDiagonal();
  state.time_micros = time_micros;
  state.center_z = box.center_z;
  state.length = box.length;
  state.width = box.width;
  state.height = box.height;
  return state;
}

void MotionFilter::advance(TrackState& state, std::int64_t time_micros) const {
  if (time_micros <= state.time_micros) return;
  const auto elapsed_micros =
      static_cast<double>(time_micros - state.time_micros);
  const double dt_seconds =
      elapsed_micros / static_cast<double>(kMicrosPerSecond);
  predict(state, dt_seconds);
  state.time_micros = time_micros;
}

void MotionFilter::predict(TrackState& state, double dt_seconds) const {
  const double yaw = state.mean(kStateYaw);
  const double speed = state.mean(kStateSpeed);
  const double yaw_rate = state.mean(kStateYawRate);
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  const double yaw_next = yaw + yaw_rate * dt_seconds;

  // F is the Jacobian of the motion below; yaw always advances by
  // yaw_rate * dt, and only the position formula depends on the branch.
  StateMatrix motion_jacobian = StateMatrix::Identity();
  motion_jacobian(kStateYaw, kStateYawRate) = dt_seconds;

  if (std::abs(yaw_rate) < kStraightLineYawRate) {
    state.mean(kStateX) += speed * cos_yaw * dt_seconds;
    state.mean(kStateY) += speed * sin_yaw * dt_seconds;
    motion_jacobian(kStateX, kStateYaw) = -speed * sin_yaw * dt_seconds;
    motion_jacobian(kStateX, kStateSpeed) = cos_yaw * dt_seconds;
    motion_jacobian(kStateY, kStateYaw) = speed * cos_yaw * dt_seconds;
    motion_jacobian(kStateY, kStateSpeed) = sin_yaw * dt_seconds;
  } else {
    // The turning path of radius r = speed / yaw_rate moves x by
    // r (sin yaw' - sin yaw) and y by r (cos yaw - cos yaw').
    const double sin_yaw_next = std::sin(yaw_next);
    const double cos_yaw_next = std::cos(yaw_next);
    const double radius = speed / yaw_rate;
    const double sin_change = sin_yaw_next - sin_yaw;
    const double cos_change = cos_yaw - cos_yaw_next;
    state.mean(kStateX) += radius * sin_change;
    state.mean(kStateY) += radius * cos_change;
    motion_jacobian(kStateX, kStateYaw) = radius * (cos_yaw_next - cos_yaw);
    motion_jacobian(kStateX, kStateSpeed) = sin_change / yaw_rate;
    motion_jacobian(kStateX, kStateYawRate) =
        radius * dt_seconds * cos_yaw_next - radius * sin_change / yaw_rate;
    motion_jacobian(kStateY, kStateYaw) = radius * sin_change;
    motion_jacobian(kStateY, kStateSpeed) = cos_change / yaw_rate;
    motion_jacobian(kStateY, kStateYawRate) =
        radius * dt_seconds * sin_yaw_next - radius * cos_change / yaw_rate;
  }
  state.mean(kStateYaw) = wrap_angle(yaw_next);

  // Q = sa^2 g g^T + sy^2 h h^T, g = (dt^2/2 cos yaw, dt^2/2 sin yaw, 0,
  // dt, 0) and h = (0, 0, dt^2/2, 0, dt).
  const double half_dt_squared = 0.5 * dt_seconds * dt_seconds;
  StateVector acceleration_direction = StateVector::Zero();
  acceleration_direction(kStateX) = half_dt_squared * cos_yaw;
  acceleration_direction(kStateY) = half_dt_squared * sin_yaw;
  acceleration_direction(kStateSpeed) = dt_seconds;
  StateVector yaw_acceleration_direction = StateVector::Zero();
  yaw_acceleration_direction(kStateYaw) = half_dt_squared;
  yaw_acceleration_direction(kStateYawRate) = dt_seconds;

  const double acceleration_sigma = noise_.sigma_acceleration;
  const double yaw_acceleration_sigma = noise_.sigma_yaw_acceleration;
  const StateMatrix process_noise =
      acceleration_sigma * acceleration_sigma * acceleration_direction *
          acceleration_direction.transpose() +
      yaw_acceleration_sigma * yaw_acceleration_sigma *
          yaw_acceleration_direction * yaw_acceleration_direction.transpose();

  // P = F P F^T + Q.
  state.covariance =
      motion_jacobian * state.covariance * motion_jacobian.transpose() +
      process_noise;
}

double MotionFilter::distance_squared(const TrackState& state, Box box) const {
  const Eigen::Vector3d residual = innovation(state, box);
  // S = H P H^T + R, and H picks x, y and yaw, so H P H^T is P's top-left
  // 3x3 block; d^2 = y^T S^-1 y, solved rather than inverted.
  const Eigen::Matrix3d innovation_covariance =
      state.covariance.topLeftCorner<kMeasuredSize, kMeasuredSize>() +
      measurement_noise_;
  return residual.transpose() * innovation_covariance.ldlt().solve(residual);
}

void MotionFilter::update(TrackState& state, Box box) const {
  const Eigen::Vector3d residual = innovation(state, box);
  const Eigen::Matrix3d innovation_covariance =
      state.covariance.topLeftCorner<kMeasuredSize, kMeasuredSize>() +
      measurement_noise_;
  // K = P H^T S^-1, and P H^T is the first three columns of P.
  const Eigen::Matrix<double, kStateSize, kMeasuredSize> kalman_gain =
      state.covariance.leftCols<kMeasuredSize>() *
      innovation_covariance.inverse();

  state.mean += kalman_gain * residual;
  state.mean(kStateYaw) = wrap_angle(state.mean(kStateYaw));

  // Joseph form P = (I - K H) P (I - K H)^T + K R K^T keeps P symmetric and
  // positive definite; I - K H is I with K subtracted from its first columns.
  StateMatrix joseph_factor = StateMatrix::Identity();
  joseph_factor.leftCols<kMeasuredSize>() -= kalman_gain;
  state.covariance =
      joseph_factor * state.covariance * joseph_factor.transpose() +
      kalman_gain * measurement_noise_ * kalman_gain.transpose();

  state.center_z += kSizeSmoothing * (box.center_z - state.center_z);
  state.length += kSizeSmoothing * (box.length - state.length);
  state.width += kSizeSmoothing * (box.width - state.width);
  state.height += kSizeSmoothing * (box.height - state.height);
}
