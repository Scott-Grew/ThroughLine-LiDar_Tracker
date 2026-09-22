// Checks that a replay is reproducible from its seed, that the dropout
// setting reaches the tracker, and that the segment log round-trips.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "replay.hpp"
#include "synthetic.hpp"

namespace {

// True when both runs hold the same track ids, means and covariances in
// each frame, compared exactly.
bool tracks_identical(const std::vector<std::vector<Track>>& first,
                      const std::vector<std::vector<Track>>& second) {
  if (first.size() != second.size()) return false;
  for (std::size_t frame_index = 0; frame_index < first.size(); ++frame_index) {
    if (first[frame_index].size() != second[frame_index].size()) return false;
    for (std::size_t track_index = 0; track_index < first[frame_index].size();
         ++track_index) {
      const Track& first_track = first[frame_index][track_index];
      const Track& second_track = second[frame_index][track_index];
      if (first_track.track_id != second_track.track_id) return false;
      if (first_track.state.mean != second_track.state.mean) return false;
      if (first_track.state.covariance != second_track.state.covariance)
        return false;
    }
  }
  return true;
}

// Replays the segment at the given dropout and seed and returns the
// confirmed tracks per frame.
std::vector<std::vector<Track>> run_replay(const SegmentLog& segment,
                                           double dropout, std::uint64_t seed) {
  ReplaySettings settings;
  settings.tracker.noise = kTestNoise;
  settings.seed = seed;
  settings.perturbation.dropout_probability = dropout;
  Replay replay(segment, settings);
  replay.run();
  return replay.confirmed_tracks_per_frame();
}

}  // namespace

// Writes a segment to a temporary log and reads it back, comparing the
// frame count and the first and last box exactly.
TEST_CASE("segment log round-trips") {
  const SegmentLog segment = make_synthetic_segment(30, 5, 42);

  const std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() / "tracker_test_segment.trklog";
  write_segment_log(temp_path.string(), segment);
  const SegmentLog read_back = read_segment_log(temp_path.string());
  std::filesystem::remove(temp_path);

  REQUIRE(read_back.frames.size() == segment.frames.size());
  const Box& original_first_box =
      segment.frames.front().ground_truth.front().box;
  const Box& read_first_box = read_back.frames.front().ground_truth.front().box;
  REQUIRE(read_first_box.center_x == original_first_box.center_x);
  REQUIRE(read_first_box.center_y == original_first_box.center_y);
  REQUIRE(read_first_box.yaw == original_first_box.yaw);
  const Box& original_last_box = segment.frames.back().ground_truth.back().box;
  const Box& read_last_box = read_back.frames.back().ground_truth.back().box;
  REQUIRE(read_last_box.center_x == original_last_box.center_x);
  REQUIRE(read_last_box.center_y == original_last_box.center_y);
  REQUIRE(read_last_box.yaw == original_last_box.yaw);
}

// Two replays of one segment at dropout 0.3 with the same seed must agree
// exactly, frame for frame.
TEST_CASE("same seed gives identical tracks") {
  const SegmentLog segment = make_synthetic_segment(30, 5, 42);
  REQUIRE(tracks_identical(run_replay(segment, 0.3, 7),
                           run_replay(segment, 0.3, 7)));
}

// Replays at dropout 0, 0.5 and 1 must confirm fewer track instances as
// dropout rises, which a run compared only with itself could not show.
TEST_CASE("dropout setting reaches the tracker") {
  const SegmentLog segment = make_synthetic_segment(30, 5, 42);
  const auto confirmed_count = [&](double dropout) {
    std::size_t confirmed_instance_count = 0;
    for (const std::vector<Track>& frame_tracks :
         run_replay(segment, dropout, 3))
      confirmed_instance_count += frame_tracks.size();
    return confirmed_instance_count;
  };

  const std::size_t without_dropout = confirmed_count(0.0);
  const std::size_t with_dropout = confirmed_count(0.5);
  const std::size_t with_everything_dropped = confirmed_count(1.0);

  INFO("confirmed track instances without dropout: " << without_dropout);
  INFO("confirmed track instances at half dropout: " << with_dropout);
  REQUIRE(without_dropout > 0);
  REQUIRE(with_dropout < without_dropout);
  REQUIRE(with_everything_dropped == 0);
}
