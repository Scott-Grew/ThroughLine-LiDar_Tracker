// Checks that Tracker::step makes the same number of heap allocations per
// frame once a scene is steady, by counting calls to global operator new.

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdlib>
#include <new>

#include "synthetic.hpp"
#include "tracker.hpp"

namespace {

// A six-object scene long enough to reach a steady state by frame 20.
constexpr int kFrameCount = 40;
constexpr int kObjectCount = 6;
constexpr std::uint64_t kSceneSeed = 11;

std::size_t g_allocation_count = 0;
}  // namespace

// Replacing global operator new here counts allocations for the whole test
// binary, not only this file.
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

// Compares the allocations step() makes at frame 20 and at frame 40 of a
// six-object scene with no dropout, where no track starts or ends between.
TEST_CASE("step allocation count is steady") {
  const SegmentLog segment =
      make_synthetic_segment(kFrameCount, kObjectCount, kSceneSeed);

  TrackerSettings settings;
  settings.noise = kTestNoise;
  Tracker tracker(settings);

  std::size_t allocations_at_frame_20 = 0;
  std::size_t allocations_at_frame_40 = 0;

  for (std::size_t frame_index = 0; frame_index < segment.frames.size();
       ++frame_index) {
    const Frame& frame = segment.frames[frame_index];
    std::vector<Detection> detections;
    detections.reserve(frame.ground_truth.size());
    for (const GroundTruthBox& ground_truth_box : frame.ground_truth)
      detections.push_back(
          Detection{ground_truth_box.object_class, ground_truth_box.box, 1.0f});

    const std::size_t allocation_count_before = g_allocation_count;
    tracker.step(frame.capture_time_micros, detections, frame.vehicle_to_world);
    const std::size_t allocation_count_after = g_allocation_count;

    if (frame_index == 19)
      allocations_at_frame_20 =
          allocation_count_after - allocation_count_before;
    if (frame_index == 39)
      allocations_at_frame_40 =
          allocation_count_after - allocation_count_before;
  }

  INFO("allocations at frame 20: " << allocations_at_frame_20);
  INFO("allocations at frame 40: " << allocations_at_frame_40);
  REQUIRE(allocations_at_frame_20 == allocations_at_frame_40);
}
