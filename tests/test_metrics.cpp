#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "metrics.hpp"

// This file checks the CLEAR bookkeeping against a scene small enough to count by hand: three
// objects over ten frames, with one deliberate identity switch, one deliberate miss, one
// deliberate spurious track, and the fragmentation that switching an object's track on and off is
// supposed to produce. If the running totals do not match what a person counting the same ten
// frames on paper would get, the bookkeeping itself is wrong, independent of any real data.

namespace {

Track make_track(std::uint64_t track_id, double x, double y) {
  Track track{};
  track.track_id = track_id;
  track.object_class = ObjectClass::Vehicle;
  track.status = TrackStatus::Confirmed;
  track.state.mean << x, y, 0.0, 0.0, 0.0;
  track.state.covariance.setIdentity();
  track.state.center_z = 0.0;
  track.state.length = 4.0;
  track.state.width = 2.0;
  track.state.height = 1.5;
  track.last_update_micros = 0;
  track.state_time_micros = 0;
  track.hits = 10;
  track.consecutive_misses = 0;
  return track;
}

GroundTruthBox make_ground_truth(std::uint64_t object_id, double x, double y) {
  GroundTruthBox truth{};
  truth.object_id = object_id;
  truth.object_class = ObjectClass::Vehicle;
  truth.box = Box{x, y, 0.0, 4.0, 2.0, 1.5, 0.0};
  truth.lidar_points_in_box = 50;
  truth.tracking_difficulty = 1;
  return truth;
}

}

TEST_CASE("mota matches hand-computed values on a synthetic scene") {
  TrackingMetrics metrics;
  const Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

  for (int frame = 1; frame <= 10; ++frame) {
    const std::vector<GroundTruthBox> ground_truth = {
        make_ground_truth(1, 0.0, 0.0),
        make_ground_truth(2, 10.0, 0.0),
        make_ground_truth(3, 20.0, 0.0),
    };

    std::vector<Track> tracks;
    tracks.push_back(make_track(1, 0.0, 0.0));
    tracks.push_back(make_track(frame < 5 ? 2 : 99, 10.0, 0.0));
    if (frame != 7) tracks.push_back(make_track(3, 20.0, 0.0));
    if (frame == 8) tracks.push_back(make_track(55, 500.0, 500.0));

    metrics.update(tracks, ground_truth, pose);
  }

  const ClassMetrics& vehicle_metrics = metrics.per_class().at(ObjectClass::Vehicle);
  REQUIRE(vehicle_metrics.mota() == Catch::Approx(0.9));
  REQUIRE(vehicle_metrics.id_switches == 1);
  REQUIRE(vehicle_metrics.misses == 1);
  REQUIRE(vehicle_metrics.false_positives == 1);
  REQUIRE(vehicle_metrics.fragmentations == 1);
}
