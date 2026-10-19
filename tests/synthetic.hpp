// Test-only helpers, a simulated constant turn rate object and the shared
// test constants, used by the filter test.

#pragma once

#include <cmath>
#include "filter.hpp"

// Values shared by the tests, the filter noise, the simulated vehicle's box
// in metres, and the 10 Hz frame period.
inline constexpr FilterNoise kTestNoise{.sigma_acceleration = 2.0,
                                        .sigma_yaw_acceleration = 0.5,
                                        .sigma_measurement_position = 0.1,
                                        .sigma_measurement_yaw = 0.02};
inline constexpr double kTestVehicleLength = 4.5;
inline constexpr double kTestVehicleWidth = 2.0;
inline constexpr double kTestVehicleHeight = 1.6;
inline constexpr double kTestStepSeconds = 0.1;

// A simulated object's motion in the world frame. x and y are metres, yaw
// radians, speed m/s and yaw_rate rad/s.
struct TruthMotion {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double speed = 0.0;
  double yaw_rate = 0.0;
};

// The same constant turn rate motion the filter predicts, computed with no
// shared code so the filter is checked against an independent truth.
inline void advance_constant_turn_rate(TruthMotion& motion, double seconds) {
  const double yaw_next = motion.yaw + motion.yaw_rate * seconds;
  if (std::abs(motion.yaw_rate) < kStraightLineYawRate) {
    motion.x += motion.speed * std::cos(motion.yaw) * seconds;
    motion.y += motion.speed * std::sin(motion.yaw) * seconds;
  } else {
    const double radius = motion.speed / motion.yaw_rate;
    motion.x += radius * (std::sin(yaw_next) - std::sin(motion.yaw));
    motion.y += radius * (std::cos(motion.yaw) - std::cos(yaw_next));
  }
  motion.yaw = yaw_next;
}
