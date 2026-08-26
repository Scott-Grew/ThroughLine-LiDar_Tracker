#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdlib>
#include <new>
#include "synthetic.hpp"
#include "tracker.hpp"

// This file checks that a tracker running over an ordinary scene settles down: once every track
// is confirmed and nothing is being created or destroyed, stepping the tracker one more frame
// should cost the same handful of allocations it cost the frame before, not more. Growing
// allocation counts in a scene that never changes shape is exactly what a slow leak or an
// accidental per-frame reallocation looks like before it becomes a real-time problem, so global
// new and delete are replaced here with counting versions for the length of this file.

namespace {
std::size_t g_allocation_count = 0;
}

void* operator new(std::size_t size) {
  ++g_allocation_count;
  void* pointer = std::malloc(size);
  if (pointer == nullptr) throw std::bad_alloc();
  return pointer;
}

void operator delete(void* pointer) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

TEST_CASE("tracker step allocation count does not grow across a steady scene") {
  const SegmentLog segment = make_synthetic_segment(40, 6, 11);

  TrackerSettings settings;
  settings.noise.sigma_measurement_position = 0.1;
  settings.noise.sigma_measurement_yaw = 0.02;
  Tracker tracker(settings);

  std::size_t allocations_at_frame_20 = 0;
  std::size_t allocations_at_frame_40 = 0;

  for (std::size_t frame_index = 0; frame_index < segment.frames.size(); ++frame_index) {
    const Frame& frame = segment.frames[frame_index];
    std::vector<Detection> detections;
    detections.reserve(frame.ground_truth.size());
    for (const GroundTruthBox& truth : frame.ground_truth) detections.push_back(Detection{truth.object_class, truth.box, 1.0f});

    const std::size_t allocation_count_before = g_allocation_count;
    tracker.step(frame.capture_time_micros, detections, frame.vehicle_to_world);
    const std::size_t allocation_count_after = g_allocation_count;

    if (frame_index == 19) allocations_at_frame_20 = allocation_count_after - allocation_count_before;
    if (frame_index == 39) allocations_at_frame_40 = allocation_count_after - allocation_count_before;
  }

  INFO("allocations at frame 20: " << allocations_at_frame_20);
  INFO("allocations at frame 40: " << allocations_at_frame_40);
  REQUIRE(allocations_at_frame_20 == allocations_at_frame_40);
}
