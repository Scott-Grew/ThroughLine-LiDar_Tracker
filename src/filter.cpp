#include "filter.hpp"

#include <cmath>
#include <stdexcept>

// This file is the estimator. Every tracked object owns five numbers
// - where it is (x and y), which way it faces, how fast it is going,
// and how quickly it is turning - plus a table of how wrong each of
// those might be. Nothing outside this file changes those numbers.
// The tracker calls predict when time passes and update when a new
// box arrives; the assignment code calls mahalanobis_squared to ask
// whether a box could belong to this object; the viewer draws the
// uncertainty this file produces. Speed and turn rate are never
// measured by anything - they exist only because this file infers
// them from how the box moves between frames.

namespace {

// How unsure we are about speed and turning the very first time we
// see an object. Both are guesses of zero, so the uncertainty is set
// wide enough to cover a parked car through a highway car, which lets
// the first few frames correct them freely instead of fighting a
// confident zero.
constexpr double kInitialSpeedSigma = 10.0;
constexpr double kInitialYawRateSigma = 1.0;

// Below this turn rate an object counts as going straight. The
// turning formula divides by the turn rate, so without this cutoff
// every straight-driving car would divide by zero and poison its
// track with invalid numbers. Straight driving is the common case, so
// this branch runs constantly.
constexpr double kStraightLineYawRate = 1e-4;

// How much of a newly measured box size to believe each frame. A
// car's length does not change, so size is smoothed separately rather
// than being estimated like position and speed.
constexpr double kSizeSmoothing = 0.3;

// Says which of the five numbers the sensor actually sees: position
// and facing, never speed or turn rate. Everything that compares a
// prediction to a measurement goes through this.
Eigen::Matrix<double, 3, 5> measurement_jacobian() {
  Eigen::Matrix<double, 3, 5> matrix =
      Eigen::Matrix<double, 3, 5>::Zero();
  matrix(0, 0) = 1.0;
  matrix(1, 1) = 1.0;
  matrix(2, 2) = 1.0;
  return matrix;
}

// How much the box itself jitters frame to frame. These values are
// measured from the data by the staging script, not invented here.
// They decide how much the filter trusts each new box.
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

// Folds an angle back into the range minus half a turn to plus half a
// turn. Facing angles wrap around, so two cars pointing almost the
// same way can differ by nearly a full turn on paper. Skip this
// anywhere and the filter sees an enormous error where there is none,
// and throws away a good track. It is called after every place an
// angle is added or subtracted.
double wrap_angle(double radians) {
  return std::remainder(radians, 2.0 * M_PI);
}

// Builds the starting five numbers for an object the tracker has just
// seen for the first time. Refuses a sensor with no jitter at all,
// because that would claim the first box is perfect and leave the
// object with zero uncertainty, which every later calculation would
// divide by. Position and facing are copied straight from the box
// because that is exactly what was measured. Speed and turn rate
// start at zero, but marked as very unsure, so the next few frames
// can move them to the truth. The size of the box is carried along
// untouched.
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

// Moves an object forward in time without looking at any new data,
// and widens its uncertainty because a guess is not a sighting. This
// is what lets a track survive a frame where the object was missed,
// and it is what draws the predicted path on screen. The jacobian
// carries the old uncertainty through the motion; the process noise
// added at the end admits that real cars speed up and brake while
// this motion model assumes they do not. That unmodelled acceleration
// acts along the way the object is already facing, so it pushes
// position and speed together rather than as two separate guesses -
// the process noise below is built from that single push, not from
// independent doubts about each number.
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

// The plain disagreement between where a box is and where this object
// was predicted to be. The facing part is wrapped, because the
// shortest way round is the only meaningful answer.
Eigen::Vector3d innovation(const TrackState& state,
                           const Box& measurement) {
  Eigen::Vector3d difference;
  difference(0) = measurement.center_x - state.mean(0);
  difference(1) = measurement.center_y - state.mean(1);
  difference(2) = wrap_angle(measurement.yaw - state.mean(2));
  return difference;
}

// How much disagreement would be normal, given both how unsure the
// prediction is and how much the sensor jitters. Without this there
// is no way to tell a small error from a large one.
Eigen::Matrix3d innovation_covariance(const TrackState& state,
                                      const FilterNoise& noise) {
  const Eigen::Matrix<double, 3, 5> observation =
      measurement_jacobian();
  return observation * state.covariance * observation.transpose() +
         measurement_noise(noise);
}

// Turns a disagreement in metres into a disagreement in units of how
// surprised we should be. Two metres is enormous for an object we
// have watched for a second and nothing at all for one we just found.
// The assignment code uses this number to decide which box belongs to
// which object, and to refuse a pairing that is too far-fetched, so
// this is the gatekeeper against identity swaps.
double mahalanobis_squared(const TrackState& state,
                           const Box& measurement,
                           const FilterNoise& noise) {
  const Eigen::Vector3d difference = innovation(state, measurement);
  const Eigen::Matrix3d covariance =
      innovation_covariance(state, noise);
  return difference.transpose() * covariance.ldlt().solve(difference);
}

// Folds a new box into the object, pulling the five numbers toward
// what was just seen. How far they move is decided by which is
// currently more trustworthy, the prediction or the sensor - nothing
// is hand-tuned here. Because position and speed have become linked
// by watching the object move, correcting the position also corrects
// the speed, which is how a quantity the sensor never measures gets
// learned at all. The long covariance line is the numerically safe
// form of the update; the short textbook version drifts until the
// uncertainty stops being valid and the gatekeeper above starts
// making nonsense decisions. Box size is not estimated, only
// smoothed.
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
