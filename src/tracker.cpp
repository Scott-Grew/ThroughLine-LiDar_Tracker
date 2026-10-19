// Runs the track life cycle once per frame: predict, match, update, start
// and delete. main.cpp reads back the confirmed tracks.

#include "tracker.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <dlib/optimization/max_cost_assignment.h>
#include "filter.hpp"
#include "types.hpp"

namespace {

// Trajectory points kept per track, oldest dropped first.
constexpr std::size_t kHistoryLength = 30;

// dlib solves over integers. Costs are squared Mahalanobis distances at
// most the gate, so this scale keeps six decimal places of resolution.
constexpr double kCostScale = 1e6;

// Reward each gated pair earns on top of its cost, bigger than any total
// the scaled costs reach, so a match always beats leaving a pair out.
constexpr std::int64_t kPairReward = 1'000'000'000'000;

// The frame's boxes moved from the vehicle frame to the world frame.
std::vector<Detection> world_detections(const Frame& frame) {
  const Eigen::Matrix4d& vehicle_to_world = frame.vehicle_to_world;
  const double ego_yaw = vehicle_yaw(vehicle_to_world);
  std::vector<Detection> detections;
  detections.reserve(frame.detections.size());
  for (const Detection& detection : frame.detections) {
    const Box& box = detection.box;
    const Eigen::Vector4d local_center(box.center_x, box.center_y, box.center_z,
                                       1.0);
    const Eigen::Vector4d world_center = vehicle_to_world * local_center;
    Box world_box = box;
    world_box.center_x = world_center.x();
    world_box.center_y = world_center.y();
    world_box.center_z = world_center.z();
    world_box.yaw = wrap_angle(box.yaw + ego_yaw);
    detections.push_back(
        {.object_class = detection.object_class, .box = world_box});
  }
  return detections;
}

void append_history(Track& track) {
  track.history.push_back(track.state.position());
  if (track.history.size() > kHistoryLength)
    track.history.erase(track.history.begin());
}

}  // namespace

Assignment assign(const Eigen::MatrixXd& cost, double gate) {
  const Eigen::Index row_count = cost.rows();
  const Eigen::Index column_count = cost.cols();
  const Eigen::Index side = std::max(row_count, column_count);
  // Padding rows and columns stay at zero reward, so dlib never prefers
  // them over a real gated pair.
  dlib::matrix<std::int64_t> reward(side, side);
  reward = 0;
  for (Eigen::Index row = 0; row < row_count; ++row)
    for (Eigen::Index column = 0; column < column_count; ++column)
      if (cost(row, column) <= gate)
        reward(row, column) = kPairReward - static_cast<std::int64_t>(
                                                cost(row, column) * kCostScale);

  Assignment assignment;
  if (side == 0) return assignment;
  std::vector<bool> column_matched(column_count, false);
  const auto column_of_row = dlib::max_cost_assignment(reward);
  for (Eigen::Index row = 0; row < row_count; ++row) {
    const Eigen::Index column = column_of_row[row];
    const bool matched = column < column_count && cost(row, column) <= gate;
    if (!matched) {
      assignment.unmatched_rows.push_back(static_cast<int>(row));
      continue;
    }
    assignment.pairs.emplace_back(static_cast<int>(row),
                                  static_cast<int>(column));
    column_matched[column] = true;
  }
  for (Eigen::Index column = 0; column < column_count; ++column)
    if (!column_matched[column])
      assignment.unmatched_columns.push_back(static_cast<int>(column));
  return assignment;
}

Tracker::Tracker(TrackerSettings settings)
    : settings_(settings), filter_(settings.noise) {}

void Tracker::step(const Frame& frame) {
  const std::int64_t time_micros = frame.capture_time_micros;
  for (Track& track : tracks_) filter_.advance(track.state, time_micros);

  const std::vector<Detection> detections = world_detections(frame);
  const Eigen::MatrixXd cost = costs(detections);
  const Assignment assignment = assign(cost, settings_.gate_chi_squared);

  for (const auto& [track_index, detection_index] : assignment.pairs)
    update_matched(tracks_[track_index], detections[detection_index].box);
  for (const int track_index : assignment.unmatched_rows)
    tracks_[track_index].consecutive_misses += 1;
  for (const int detection_index : assignment.unmatched_columns)
    tracks_.push_back(new_track(detections[detection_index], time_micros));

  for (Track& track : tracks_) track.status = next_status(track);
  std::erase_if(tracks_, [](const Track& track) {
    return track.status == TrackStatus::Dead;
  });
}

std::vector<Track> Tracker::confirmed_at(std::int64_t time_micros) const {
  std::vector<Track> confirmed;
  for (const Track& track : tracks_) {
    const bool reported = track.status == TrackStatus::Confirmed ||
                          track.status == TrackStatus::Coasting;
    if (!reported) continue;
    Track predicted = track;
    filter_.advance(predicted.state, time_micros);
    confirmed.push_back(std::move(predicted));
  }
  return confirmed;
}

// A class mismatch costs infinity so assign() gates it out, which lets
// one matrix stand in for one matrix per class.
Eigen::MatrixXd Tracker::costs(const std::vector<Detection>& detections) const {
  constexpr double kNoMatch = std::numeric_limits<double>::infinity();
  const auto track_count = static_cast<Eigen::Index>(tracks_.size());
  const auto detection_count = static_cast<Eigen::Index>(detections.size());
  Eigen::MatrixXd cost(track_count, detection_count);
  for (Eigen::Index row = 0; row < track_count; ++row) {
    const Track& track = tracks_[row];
    for (Eigen::Index column = 0; column < detection_count; ++column) {
      const Detection& detection = detections[column];
      const bool same_class = track.object_class == detection.object_class;
      cost(row, column) =
          same_class ? filter_.distance_squared(track.state, detection.box)
                     : kNoMatch;
    }
  }
  return cost;
}

Track Tracker::new_track(const Detection& detection, std::int64_t time_micros) {
  Track track;
  track.track_id = next_track_id_++;
  track.object_class = detection.object_class;
  track.state = filter_.start_state(detection.box, time_micros);
  track.hits = 1;
  append_history(track);
  return track;
}

void Tracker::update_matched(Track& track, const Box& box) const {
  filter_.update(track.state, box);
  track.hits += 1;
  track.consecutive_misses = 0;
  append_history(track);
}

// The whole track life cycle: confirmation on a clean hit, death from
// tentative on any miss, death from a run of misses, otherwise coasting.
TrackStatus Tracker::next_status(const Track& track) const {
  if (track.consecutive_misses == 0)
    return track.hits >= settings_.hits_to_confirm ? TrackStatus::Confirmed
                                                   : TrackStatus::Tentative;
  if (track.status == TrackStatus::Tentative) return TrackStatus::Dead;
  if (track.consecutive_misses >= settings_.misses_to_delete)
    return TrackStatus::Dead;
  return TrackStatus::Coasting;
}
