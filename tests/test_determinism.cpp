#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "replay.hpp"
#include "synthetic.hpp"

// This file is the guarantee the whole replay design rests on: the
// same segment and the same seed produce the same tracks, frame for
// frame, and a perturbation setting actually reaches the tracker. It
// also checks the log format round-trips exactly.

namespace {

bool tracks_identical(const std::vector<std::vector<Track>>& first,
                      const std::vector<std::vector<Track>>& second) {
  if (first.size() != second.size()) return false;
  for (std::size_t frame_index = 0; frame_index < first.size();
       ++frame_index) {
    if (first[frame_index].size() != second[frame_index].size())
      return false;
    for (std::size_t track_index = 0;
         track_index < first[frame_index].size(); ++track_index) {
      const Track& left = first[frame_index][track_index];
      const Track& right = second[frame_index][track_index];
      if (left.track_id != right.track_id) return false;
      if (left.state.mean != right.state.mean) return false;
      if (left.state.covariance != right.state.covariance)
        return false;
    }
  }
  return true;
}

std::vector<std::vector<Track>> run_replay(const SegmentLog& segment,
                                           double dropout,
                                           std::uint64_t seed) {
  ReplaySettings settings;
  settings.tracker.noise = FilterNoise{2.0, 0.5, 0.1, 0.02};
  settings.seed = seed;
  settings.perturbation.dropout_probability = dropout;
  Replay replay(segment, settings);
  replay.run();
  return replay.confirmed_tracks_per_frame();
}

}  // namespace

TEST_CASE(
    "tracker output is byte-identical across two runs with the same "
    "seed") {
  const SegmentLog segment = make_synthetic_segment(30, 5, 42);

  const std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() /
      "tracker_test_segment.trklog";
  write_segment_log(temp_path.string(), segment);
  const SegmentLog read_back = read_segment_log(temp_path.string());
  std::filesystem::remove(temp_path);

  REQUIRE(read_back.frames.size() == segment.frames.size());
  const Box& original_first_box =
      segment.frames.front().ground_truth.front().box;
  const Box& read_first_box =
      read_back.frames.front().ground_truth.front().box;
  REQUIRE(read_first_box.center_x == original_first_box.center_x);
  REQUIRE(read_first_box.center_y == original_first_box.center_y);
  REQUIRE(read_first_box.yaw == original_first_box.yaw);
  const Box& original_last_box =
      segment.frames.back().ground_truth.back().box;
  const Box& read_last_box =
      read_back.frames.back().ground_truth.back().box;
  REQUIRE(read_last_box.center_x == original_last_box.center_x);
  REQUIRE(read_last_box.center_y == original_last_box.center_y);
  REQUIRE(read_last_box.yaw == original_last_box.yaw);

  REQUIRE(tracks_identical(run_replay(segment, 0.3, 7),
                           run_replay(segment, 0.3, 7)));
}

// A perturbation asked for in the settings has to actually reach the
// tracker. Comparing a run to itself proves nothing about whether a
// setting does anything, so this compares runs that were asked for
// different things and insists they disagree.
TEST_CASE("a dropout setting changes what the tracker sees") {
  const SegmentLog segment = make_synthetic_segment(30, 5, 42);
  const auto confirmed_count = [&](double dropout) {
    std::size_t total = 0;
    for (const std::vector<Track>& frame_tracks :
         run_replay(segment, dropout, 3))
      total += frame_tracks.size();
    return total;
  };

  const std::size_t without_dropout = confirmed_count(0.0);
  const std::size_t with_dropout = confirmed_count(0.5);
  const std::size_t with_everything_dropped = confirmed_count(1.0);

  INFO("confirmed track instances without dropout: "
       << without_dropout);
  INFO("confirmed track instances at half dropout: " << with_dropout);
  REQUIRE(without_dropout > 0);
  REQUIRE(with_dropout < without_dropout);
  REQUIRE(with_everything_dropped == 0);
}
