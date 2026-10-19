// Checks the filter's reported covariance against its real error using
// NEES, the squared error in units of the filter's own covariance.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>
#include "filter.hpp"
#include "synthetic.hpp"

namespace {

// Run sizes are chosen. The band is the 95% interval for the mean of 50
// NEES values with 5 degrees of freedom, 5 +/- 1.96 * sqrt(2 * 5 / 50).
constexpr int kStepCount = 200;
constexpr int kWarmupSteps = 20;
constexpr int kRunCount = 50;
constexpr double kBandLow = 4.12;
constexpr double kBandHigh = 5.88;
constexpr double kRequiredFractionInsideBand = 0.90;

std::vector<double> nees_per_step_for_seed(std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::normal_distribution<double> acceleration_draw(
      0.0, kVehiclePrior.acceleration_sigma);
  std::normal_distribution<double> yaw_acceleration_draw(
      0.0, kTestNoise.sigma_yaw_acceleration);
  std::normal_distribution<double> position_measurement_noise(
      0.0, kTestNoise.sigma_measurement_position);
  std::normal_distribution<double> yaw_measurement_noise(
      0.0, kTestNoise.sigma_measurement_yaw);

  TruthMotion truth{0.0, 0.0, 0.0, 8.0, 0.05};

  const MotionFilter filter(kTestNoise);

  const Box first_measurement{
      truth.x + position_measurement_noise(generator),
      truth.y + position_measurement_noise(generator),
      0.0,
      kTestVehicleLength,
      kTestVehicleWidth,
      kTestVehicleHeight,
      wrap_angle(truth.yaw + yaw_measurement_noise(generator))};
  const Detection first_detection{.object_class = ObjectClass::Vehicle,
                                  .box = first_measurement};
  TrackState state = filter.start_state(first_detection);

  std::vector<double> nees_per_step(kStepCount, 0.0);

  for (int step = 1; step <= kStepCount; ++step) {
    const double yaw_before_step = truth.yaw;

    advance_constant_turn_rate(truth, kTestStepSeconds);

    const double acceleration = acceleration_draw(generator);
    const double yaw_acceleration = yaw_acceleration_draw(generator);
    truth.speed += acceleration * kTestStepSeconds;
    truth.yaw_rate += yaw_acceleration * kTestStepSeconds;
    truth.x += 0.5 * acceleration * kTestStepSeconds * kTestStepSeconds *
               std::cos(yaw_before_step);
    truth.y += 0.5 * acceleration * kTestStepSeconds * kTestStepSeconds *
               std::sin(yaw_before_step);
    truth.yaw += 0.5 * yaw_acceleration * kTestStepSeconds * kTestStepSeconds;
    truth.yaw = wrap_angle(truth.yaw);

    filter.predict(state, kTestStepSeconds);

    const Box measurement{
        truth.x + position_measurement_noise(generator),
        truth.y + position_measurement_noise(generator),
        0.0,
        kTestVehicleLength,
        kTestVehicleWidth,
        kTestVehicleHeight,
        wrap_angle(truth.yaw + yaw_measurement_noise(generator))};
    filter.update(state, measurement);

    StateVector truth_vector;
    truth_vector << truth.x, truth.y, truth.yaw, truth.speed, truth.yaw_rate;
    StateVector error = state.mean - truth_vector;
    error(kStateYaw) = wrap_angle(error(kStateYaw));
    // NEES is e^T P^-1 e for error e and filter covariance P.
    nees_per_step[step - 1] =
        error.transpose() * state.covariance.ldlt().solve(error);
  }

  return nees_per_step;
}

// Perturbation size for the central differences that recover the motion
// Jacobian numerically.
constexpr double kFiniteDifferenceStep = 1e-6;

TrackState make_motion_test_state(double yaw_rate) {
  TrackState state;
  state.mean << 3.0, -2.0, 0.7, 8.0, yaw_rate;
  state.covariance = StateMatrix::Identity();
  state.time_micros = 0;
  state.acceleration_sigma = kVehiclePrior.acceleration_sigma;
  state.center_z = 0.0;
  state.length = kTestVehicleLength;
  state.width = kTestVehicleWidth;
  state.height = kTestVehicleHeight;
  return state;
}

StateMatrix difference_jacobian(const TrackState& state, double dt_seconds) {
  const MotionFilter filter(kTestNoise);
  StateMatrix numeric_jacobian;
  for (int column = 0; column < kStateSize; ++column) {
    TrackState state_plus = state;
    TrackState state_minus = state;
    state_plus.mean(column) += kFiniteDifferenceStep;
    state_minus.mean(column) -= kFiniteDifferenceStep;
    filter.predict(state_plus, dt_seconds);
    filter.predict(state_minus, dt_seconds);
    StateVector mean_difference = state_plus.mean - state_minus.mean;
    mean_difference(kStateYaw) =
        wrap_angle(state_plus.mean(kStateYaw) - state_minus.mean(kStateYaw));
    numeric_jacobian.col(column) =
        mean_difference / (2.0 * kFiniteDifferenceStep);
  }
  return numeric_jacobian;
}

void check_motion_jacobian_matches_covariance(double yaw_rate) {
  const TrackState state = make_motion_test_state(yaw_rate);
  const double dt_seconds = kTestStepSeconds;
  const MotionFilter filter(kTestNoise);

  const StateMatrix numeric_jacobian = difference_jacobian(state, dt_seconds);

  TrackState state_identity = state;
  filter.predict(state_identity, dt_seconds);
  const StateMatrix covariance_from_identity = state_identity.covariance;

  TrackState state_zero = state;
  state_zero.covariance = StateMatrix::Zero();
  filter.predict(state_zero, dt_seconds);
  const StateMatrix process_noise = state_zero.covariance;

  const double jacobian_error_norm =
      (numeric_jacobian * numeric_jacobian.transpose() -
       (covariance_from_identity - process_noise))
          .norm();
  REQUIRE(jacobian_error_norm < 1e-5);
}

}  // namespace

// Averages NEES over kRunCount seeds per step, truth driven by random
// accelerations at the filter's own sigmas; most steps must sit in the band.
TEST_CASE("filter NEES stays inside the chi-square band") {
  std::vector<double> step_sum(kStepCount, 0.0);
  for (std::uint64_t seed = 1; seed <= kRunCount; ++seed) {
    const std::vector<double> nees_per_step = nees_per_step_for_seed(seed);
    for (int step = 0; step < kStepCount; ++step)
      step_sum[step] += nees_per_step[step];
  }

  int steps_inside_band = 0;
  int steps_checked = 0;
  double mean_of_averages = 0.0;
  for (int step = kWarmupSteps; step < kStepCount; ++step) {
    const double step_average = step_sum[step] / kRunCount;
    steps_checked += 1;
    mean_of_averages += step_average;
    if (step_average >= kBandLow && step_average <= kBandHigh)
      steps_inside_band += 1;
  }
  mean_of_averages /= steps_checked;
  const double fraction_inside_band =
      static_cast<double>(steps_inside_band) / steps_checked;

  INFO("fraction of per-step averages inside the band: "
       << fraction_inside_band);
  INFO("mean of per-step averages: " << mean_of_averages);
  REQUIRE(fraction_inside_band >= kRequiredFractionInsideBand);
}

// Checks the motion Jacobian against central finite differences of
// predict's mean update. F is recovered column by column from the mean,
// then checked against the covariance identity F F^T = P'_identity - Q.
TEST_CASE("motion Jacobian matches finite differences") {
  check_motion_jacobian_matches_covariance(0.3);
  check_motion_jacobian_matches_covariance(0.0);
}
