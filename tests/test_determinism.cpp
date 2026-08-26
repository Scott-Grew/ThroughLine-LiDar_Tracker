#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include "predict.hpp"
#include "replay.hpp"
#include "synthetic.hpp"

// This file is the one guarantee the whole replay design rests on: nothing that decides what the
// tracker sees is allowed to depend on the wall clock. It checks the log format round-trips
// exactly, then checks that two runs given the same seed - one headless, one paced a thousand
// times faster than real time - land on the exact same tracks, frame for frame. If this test ever
// fails, something in replay.cpp has started reading real time to decide what the tracker sees,
// which would make every other result in this project unreproducible.

namespace {

bool tracks_identical(const std::vector<std::vector<Track>>& first, const std::vector<std::vector<Track>>& second) {
  if (first.size() != second.size()) return false;
  for (std::size_t frame_index = 0; frame_index < first.size(); ++frame_index) {
    if (first[frame_index].size() != second[frame_index].size()) return false;
    for (std::size_t track_index = 0; track_index < first[frame_index].size(); ++track_index) {
      const Track& left = first[frame_index][track_index];
      const Track& right = second[frame_index][track_index];
      if (left.track_id != right.track_id) return false;
      if (left.state.mean != right.state.mean) return false;
      if (left.state.covariance != right.state.covariance) return false;
    }
  }
  return true;
}

}

TEST_CASE("tracker output is byte-identical across two runs with the same seed") {
  const SegmentLog segment = make_synthetic_segment(30, 5, 42);

  const std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "tracker_test_segment.trklog";
  write_segment_log(temp_path.string(), segment);
  const SegmentLog read_back = read_segment_log(temp_path.string());
  std::filesystem::remove(temp_path);

  REQUIRE(read_back.frames.size() == segment.frames.size());
  const Box& original_first_box = segment.frames.front().ground_truth.front().box;
  const Box& read_first_box = read_back.frames.front().ground_truth.front().box;
  REQUIRE(read_first_box.center_x == original_first_box.center_x);
  REQUIRE(read_first_box.center_y == original_first_box.center_y);
  REQUIRE(read_first_box.yaw == original_first_box.yaw);
  const Box& original_last_box = segment.frames.back().ground_truth.back().box;
  const Box& read_last_box = read_back.frames.back().ground_truth.back().box;
  REQUIRE(read_last_box.center_x == original_last_box.center_x);
  REQUIRE(read_last_box.center_y == original_last_box.center_y);
  REQUIRE(read_last_box.yaw == original_last_box.yaw);

  const FilterNoise noise{2.0, 0.5, 0.1, 0.02};
  ConstantTurnRatePredictor predictor(noise);
  ReplaySettings headless_settings;
  headless_settings.tracker.noise = noise;
  headless_settings.headless = true;
  headless_settings.seed = 7;
  headless_settings.perturbation.dropout_probability = 0.3;

  Replay first_run(segment, headless_settings, predictor);
  SnapshotExchange first_exchange;
  LiveControls first_controls;
  first_run.run(first_exchange, first_controls);

  Replay second_run(segment, headless_settings, predictor);
  SnapshotExchange second_exchange;
  LiveControls second_controls;
  second_run.run(second_exchange, second_controls);

  REQUIRE(tracks_identical(first_run.confirmed_tracks_per_frame(), second_run.confirmed_tracks_per_frame()));

  ReplaySettings paced_settings = headless_settings;
  paced_settings.headless = false;
  paced_settings.rate = 1000.0;

  Replay paced_run(segment, paced_settings, predictor);
  SnapshotExchange paced_exchange;
  LiveControls paced_controls;
  paced_run.run(paced_exchange, paced_controls);

  REQUIRE(tracks_identical(first_run.confirmed_tracks_per_frame(), paced_run.confirmed_tracks_per_frame()));
}
