// Test-only helpers: a simulated constant turn rate object and a seeded
// synthetic segment, shared by the test files.

#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include "filter.hpp"
#include "log.hpp"

// Values shared by the tests: filter noise, the simulated vehicle's box in
// metres, and the 10 Hz frame period in both of the units the code uses.
inline constexpr FilterNoise kTestNoise{2.0, 0.5, 0.1, 0.02};
inline constexpr double kTestVehicleLength = 4.5;
inline constexpr double kTestVehicleWidth = 2.0;
inline constexpr double kTestVehicleHeight = 1.6;
inline constexpr double kTestStepSeconds = 0.1;
inline constexpr std::int64_t kTestFramePeriodMicros = 100'000;

// A simulated object's motion in the world frame: x, y in metres, yaw in
// radians, speed in m/s and yaw_rate in rad/s.
struct TruthMotion {
  double x, y, yaw, speed, yaw_rate;
};

// Advances motion by step_seconds along a constant turn rate path, the same
// noise-free model the filter predicts with. Does not wrap yaw.
inline void advance_constant_turn_rate(TruthMotion& motion,
                                       double step_seconds) {
  if (std::abs(motion.yaw_rate) < kStraightLineYawRate) {
    motion.x += motion.speed * std::cos(motion.yaw) * step_seconds;
    motion.y += motion.speed * std::sin(motion.yaw) * step_seconds;
  } else {
    const double yaw_next = motion.yaw + motion.yaw_rate * step_seconds;
    const double radius = motion.speed / motion.yaw_rate;
    motion.x += radius * (std::sin(yaw_next) - std::sin(motion.yaw));
    motion.y += radius * (std::cos(motion.yaw) - std::cos(yaw_next));
    motion.yaw = yaw_next;
  }
}

// Builds a seeded segment of vehicles on constant turn rate paths, so tests
// need no Waymo data and two builds from one seed are identical.
inline SegmentLog make_synthetic_segment(int frame_count, int object_count,
                                         std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> speed_draw(3.0, 12.0);
  std::uniform_real_distribution<double> yaw_rate_draw(0.0, 0.1);
  std::uniform_real_distribution<double> yaw_draw(-M_PI, M_PI);

  std::vector<TruthMotion> objects(object_count);
  for (int object_index = 0; object_index < object_count; ++object_index) {
    objects[object_index] = TruthMotion{
        static_cast<double>(object_index) * 50.0, 0.0, yaw_draw(generator),
        speed_draw(generator), yaw_rate_draw(generator)};
  }

  SegmentLog segment;
  segment.segment_name = "synthetic";
  segment.frames.resize(frame_count);
  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    Frame& frame = segment.frames[frame_index];
    frame.capture_time_micros =
        static_cast<std::int64_t>(frame_index) * kTestFramePeriodMicros;
    frame.vehicle_to_world = Eigen::Matrix4d::Identity();

    for (int object_index = 0; object_index < object_count; ++object_index) {
      TruthMotion& object = objects[object_index];

      GroundTruthBox ground_truth_box{};
      ground_truth_box.object_id = static_cast<std::uint64_t>(object_index + 1);
      ground_truth_box.object_class = ObjectClass::Vehicle;
      ground_truth_box.box = Box{object.x,
                                 object.y,
                                 0.0,
                                 kTestVehicleLength,
                                 kTestVehicleWidth,
                                 kTestVehicleHeight,
                                 wrap_angle(object.yaw)};
      ground_truth_box.lidar_points_in_box = 50;
      frame.ground_truth.push_back(ground_truth_box);

      advance_constant_turn_rate(object, kTestStepSeconds);
    }
  }
  return segment;
}
