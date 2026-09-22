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
      0.0, kTestNoise.sigma_acceleration);
  std::normal_distribution<double> yaw_acceleration_draw(
      0.0, kTestNoise.sigma_yaw_acceleration);
  std::normal_distribution<double> position_measurement_noise(
      0.0, kTestNoise.sigma_measurement_position);
  std::normal_distribution<double> yaw_measurement_noise(
      0.0, kTestNoise.sigma_measurement_yaw);

  TruthMotion truth{0.0, 0.0, 0.0, 8.0, 0.05};

  const FilterNoise noise = kTestNoise;

  const Box first_measurement{
      truth.x + position_measurement_noise(generator),
      truth.y + position_measurement_noise(generator),
      0.0,
      kTestVehicleLength,
      kTestVehicleWidth,
      kTestVehicleHeight,
      wrap_angle(truth.yaw + yaw_measurement_noise(generator))};
  TrackState state = initial_state(first_measurement, noise);

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

    predict(state, kTestStepSeconds, noise);

    const Box measurement{
        truth.x + position_measurement_noise(generator),
        truth.y + position_measurement_noise(generator),
        0.0,
        kTestVehicleLength,
        kTestVehicleWidth,
        kTestVehicleHeight,
        wrap_angle(truth.yaw + yaw_measurement_noise(generator))};
    update(state, measurement, noise);

    Eigen::Matrix<double, 5, 1> truth_vector;
    truth_vector << truth.x, truth.y, truth.yaw, truth.speed, truth.yaw_rate;
    Eigen::Matrix<double, 5, 1> error = state.mean - truth_vector;
    error(kYawIndex) = wrap_angle(error(kYawIndex));
    // NEES is e^T P^-1 e for error e and filter covariance P.
    nees_per_step[step - 1] =
        error.transpose() * state.covariance.ldlt().solve(error);
  }

  return nees_per_step;
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
