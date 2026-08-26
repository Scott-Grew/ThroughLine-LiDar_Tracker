#pragma once

#include <cmath>
#include <random>
#include "filter.hpp"
#include "log.hpp"

// A shared way for the tests to get a segment to run against, without any of them staging real
// Waymo data or duplicating the same synthetic scene by hand. Every object moves on a constant
// turn rate path with its own speed and turn rate, seeded so the same seed always builds the same
// segment, which is what lets the determinism and allocation tests compare two runs against a
// segment they know is identical.
inline SegmentLog make_synthetic_segment(int frame_count, int object_count, std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> speed_draw(3.0, 12.0);
  std::uniform_real_distribution<double> yaw_rate_draw(0.0, 0.1);
  std::uniform_real_distribution<double> heading_draw(-M_PI, M_PI);

  struct ObjectState {
    double x, y, yaw, speed, yaw_rate;
  };

  std::vector<ObjectState> objects(object_count);
  for (int object_index = 0; object_index < object_count; ++object_index) {
    objects[object_index] = ObjectState{static_cast<double>(object_index) * 50.0, 0.0,
                                         heading_draw(generator), speed_draw(generator), yaw_rate_draw(generator)};
  }

  constexpr double kStepSeconds = 0.1;

  SegmentLog segment;
  segment.segment_name = "synthetic";
  segment.frames.resize(frame_count);
  for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
    Frame& frame = segment.frames[frame_index];
    frame.capture_time_micros = static_cast<std::int64_t>(frame_index) * 100000;
    frame.vehicle_to_world = Eigen::Matrix4d::Identity();

    for (int object_index = 0; object_index < object_count; ++object_index) {
      ObjectState& object = objects[object_index];

      GroundTruthBox truth{};
      truth.object_id = static_cast<std::uint64_t>(object_index + 1);
      truth.object_class = ObjectClass::Vehicle;
      truth.box = Box{object.x, object.y, 0.0, 4.5, 2.0, 1.6, wrap_angle(object.yaw)};
      truth.lidar_points_in_box = 50;
      frame.ground_truth.push_back(truth);

      if (std::abs(object.yaw_rate) < 1e-4) {
        object.x += object.speed * std::cos(object.yaw) * kStepSeconds;
        object.y += object.speed * std::sin(object.yaw) * kStepSeconds;
      } else {
        const double yaw_next = object.yaw + object.yaw_rate * kStepSeconds;
        const double radius = object.speed / object.yaw_rate;
        object.x += radius * (std::sin(yaw_next) - std::sin(object.yaw));
        object.y += radius * (std::cos(object.yaw) - std::cos(yaw_next));
        object.yaw = yaw_next;
      }
    }
  }
  return segment;
}
