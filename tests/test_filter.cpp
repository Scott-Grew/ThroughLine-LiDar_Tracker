#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>
#include "filter.hpp"

// This file checks that the filter's own idea of its uncertainty is
// honest. A filter that reports a covariance can be wrong in two
// directions - too sure of itself, or needlessly unsure - and either
// one is invisible from the mean alone. NEES (the squared error
// measured in units of the filter's own uncertainty) is the standard
// way to catch both: averaged over enough independent runs, it should
// land within a known band around five, one for each of the five
// numbers the filter tracks. A truth trajectory is generated with
// genuinely random accelerations, using the exact same sigmas the
// filter's own process noise assumes, so a passing test says the
// filter's covariance is neither overclaiming nor underclaiming what
// a real object actually does between sightings.

namespace {

constexpr double kStepSeconds = 0.1;
constexpr int kStepCount = 200;
constexpr int kWarmupSteps = 20;
constexpr int kRunCount = 50;
constexpr double kBandLow = 4.12;
constexpr double kBandHigh = 5.88;

// Runs one random truth trajectory through the filter and returns the
// NEES at every step. The truth is not the filter's own motion model
// handed noise-free numbers - it is pushed by a genuinely random
// acceleration each step, along the way it is already facing, which
// is exactly the unmodelled push the filter's process noise is meant
// to account for.
std::vector<double> nees_per_step_for_seed(std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::normal_distribution<double> acceleration_draw(0.0, 2.0);
  std::normal_distribution<double> yaw_acceleration_draw(0.0, 0.5);
  std::normal_distribution<double> position_measurement_noise(0.0,
                                                              0.1);
  std::normal_distribution<double> yaw_measurement_noise(0.0, 0.02);

  double truth_x = 0.0, truth_y = 0.0, truth_yaw = 0.0;
  double truth_speed = 8.0, truth_yaw_rate = 0.05;

  const FilterNoise noise{2.0, 0.5, 0.1, 0.02};

  const Box first_measurement{
      truth_x + position_measurement_noise(generator),
      truth_y + position_measurement_noise(generator),
      0.0,
      4.5,
      2.0,
      1.6,
      wrap_angle(truth_yaw + yaw_measurement_noise(generator))};
  TrackState state = initial_state(first_measurement, noise);

  std::vector<double> nees_per_step(kStepCount, 0.0);

  for (int step = 1; step <= kStepCount; ++step) {
    const double yaw = truth_yaw;
    const double speed = truth_speed;
    const double yaw_rate = truth_yaw_rate;

    if (std::abs(yaw_rate) < 1e-4) {
      truth_x += speed * std::cos(yaw) * kStepSeconds;
      truth_y += speed * std::sin(yaw) * kStepSeconds;
    } else {
      const double yaw_next = yaw + yaw_rate * kStepSeconds;
      const double radius = speed / yaw_rate;
      truth_x += radius * (std::sin(yaw_next) - std::sin(yaw));
      truth_y += radius * (std::cos(yaw) - std::cos(yaw_next));
      truth_yaw = yaw_next;
    }

    const double acceleration = acceleration_draw(generator);
    const double yaw_acceleration = yaw_acceleration_draw(generator);
    truth_speed += acceleration * kStepSeconds;
    truth_yaw_rate += yaw_acceleration * kStepSeconds;
    truth_x += 0.5 * acceleration * kStepSeconds * kStepSeconds *
               std::cos(yaw);
    truth_y += 0.5 * acceleration * kStepSeconds * kStepSeconds *
               std::sin(yaw);
    truth_yaw += 0.5 * yaw_acceleration * kStepSeconds * kStepSeconds;
    truth_yaw = wrap_angle(truth_yaw);

    predict(state, kStepSeconds, noise);

    const Box measurement{
        truth_x + position_measurement_noise(generator),
        truth_y + position_measurement_noise(generator),
        0.0,
        4.5,
        2.0,
        1.6,
        wrap_angle(truth_yaw + yaw_measurement_noise(generator))};
    update(state, measurement, noise);

    Eigen::Matrix<double, 5, 1> truth_vector;
    truth_vector << truth_x, truth_y, truth_yaw, truth_speed,
        truth_yaw_rate;
    Eigen::Matrix<double, 5, 1> error = state.mean - truth_vector;
    error(2) = wrap_angle(error(2));
    nees_per_step[step - 1] =
        error.transpose() * state.covariance.ldlt().solve(error);
  }

  return nees_per_step;
}

}  // namespace

TEST_CASE(
    "filter NEES averaged over independent runs lies inside the "
    "chi-square band") {
  std::vector<double> step_sum(kStepCount, 0.0);
  for (std::uint64_t seed = 1; seed <= kRunCount; ++seed) {
    const std::vector<double> nees_per_step =
        nees_per_step_for_seed(seed);
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
  REQUIRE(fraction_inside_band >= 0.90);
}
