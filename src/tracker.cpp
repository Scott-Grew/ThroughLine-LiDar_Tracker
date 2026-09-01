#include "tracker.hpp"

#include <cmath>

// This file is the whole point of the project: it decides, frame by
// frame, which boxes belong to which object over time. It owns the
// list of tracks and nothing else touches that list directly. Each
// call to step hands it one frame's detections; it moves every
// existing track forward to that frame's time, works out which
// detection goes with which track separately for each object class,
// folds the matches into the filter in filter.cpp, starts new tracks
// for whatever was left over, and drops whatever has gone unseen for
// too long. Everything the viewer draws and everything the metrics in
// metrics.cpp score comes from what this file decided.

namespace {

constexpr std::size_t kHistoryLength = 30;

}

Tracker::Tracker(TrackerSettings settings) : settings_(settings) {
  tracks_.reserve(settings_.reserved_tracks);
}

// Turns a box measured in the sensor's own frame into the same box in
// the fixed world frame, using the frame's recorded vehicle pose.
// Every comparison the tracker makes between a track and a detection
// happens in world coordinates, because the vehicle itself is moving
// and a track cannot be compared against a box that is still sitting
// in a different frame's sensor frame.
Box Tracker::to_world(const Box& box,
                      const Eigen::Matrix4d& vehicle_to_world) {
  const Eigen::Vector4d local_position(box.center_x, box.center_y,
                                       box.center_z, 1.0);
  const Eigen::Vector4d world_position =
      vehicle_to_world * local_position;
  Box world_box = box;
  world_box.center_x = world_position(0);
  world_box.center_y = world_position(1);
  world_box.center_z = world_position(2);
  world_box.yaw =
      wrap_angle(box.yaw + std::atan2(vehicle_to_world(1, 0),
                                      vehicle_to_world(0, 0)));
  return world_box;
}

// Carries every track's filter state forward to the time of the frame
// that is about to be processed, so that the assignment step below
// compares each detection against where the object is predicted to be
// right now, not where it was last actually seen. Each track
// remembers, in state_time_micros, the time its own state already
// accounts for, so a track that was already updated earlier in this
// same frame is not predicted twice.
void Tracker::predict_all_to(std::int64_t capture_time_micros) {
  for (Track& track : tracks_) {
    if (track.state_time_micros >= capture_time_micros) continue;
    const double dt_seconds =
        static_cast<double>(capture_time_micros -
                            track.state_time_micros) /
        1e6;
    predict(track.state, dt_seconds, settings_.noise);
    track.state_time_micros = capture_time_micros;
  }
}

// Runs one frame through the whole life cycle: predict every track
// forward, match tracks against detections one object class at a time
// so a car is never compared against a pedestrian, fold each match
// into its track's filter, count misses against tracks nothing
// matched, start a new tentative track for every detection nothing
// claimed, and finally remove whatever has been unseen for too long.
// Matching happens per class because mixing classes into one cost
// matrix would let a cheap but wrong cross-class match crowd out the
// right one.
void Tracker::step(std::int64_t capture_time_micros,
                   const std::vector<Detection>& detections,
                   const Eigen::Matrix4d& vehicle_to_world) {
  predict_all_to(capture_time_micros);

  std::vector<Box> world_boxes;
  world_boxes.reserve(detections.size());
  for (const Detection& detection : detections)
    world_boxes.push_back(to_world(detection.box, vehicle_to_world));

  std::vector<bool> marked_for_deletion(tracks_.size(), false);
  std::vector<Track> new_tracks;

  for (ObjectClass object_class :
       {ObjectClass::Vehicle, ObjectClass::Pedestrian,
        ObjectClass::Cyclist}) {
    std::vector<int> track_indices;
    for (std::size_t index = 0; index < tracks_.size(); ++index)
      if (tracks_[index].object_class == object_class)
        track_indices.push_back(static_cast<int>(index));

    std::vector<int> detection_indices;
    for (std::size_t index = 0; index < detections.size(); ++index)
      if (detections[index].object_class == object_class)
        detection_indices.push_back(static_cast<int>(index));

    Eigen::MatrixXd& cost = cost_by_class_[object_class];
    cost.resize(track_indices.size(), detection_indices.size());
    for (int row = 0; row < static_cast<int>(track_indices.size());
         ++row)
      for (int column = 0;
           column < static_cast<int>(detection_indices.size());
           ++column)
        cost(row, column) = mahalanobis_squared(
            tracks_[track_indices[row]].state,
            world_boxes[detection_indices[column]], settings_.noise);

    const Assignment assignment = assign(
        cost, settings_.gate_chi_squared, settings_.assignment);

    for (const auto& [row, column] : assignment.pairs) {
      Track& track = tracks_[track_indices[row]];
      update(track.state, world_boxes[detection_indices[column]],
             settings_.noise);
      track.hits += 1;
      track.consecutive_misses = 0;
      track.status = track.hits >= settings_.hits_to_confirm
                         ? TrackStatus::Confirmed
                         : TrackStatus::Tentative;
      track.history.emplace_back(track.state.mean(0),
                                 track.state.mean(1));
      if (track.history.size() > kHistoryLength)
        track.history.erase(track.history.begin());
    }

    for (int unmatched_row : assignment.unmatched_rows) {
      Track& track = tracks_[track_indices[unmatched_row]];
      track.consecutive_misses += 1;
      if (track.status == TrackStatus::Tentative) {
        marked_for_deletion[track_indices[unmatched_row]] = true;
      } else {
        track.status = TrackStatus::Coasting;
        if (track.consecutive_misses >= settings_.misses_to_delete)
          marked_for_deletion[track_indices[unmatched_row]] = true;
      }
    }

    for (int unmatched_column : assignment.unmatched_columns) {
      const Box& world_box =
          world_boxes[detection_indices[unmatched_column]];
      Track new_track;
      new_track.track_id = next_track_id_++;
      new_track.object_class = object_class;
      new_track.state = initial_state(world_box, settings_.noise);
      new_track.state_time_micros = capture_time_micros;
      new_track.hits = 1;
      new_track.consecutive_misses = 0;
      new_track.status = new_track.hits >= settings_.hits_to_confirm
                             ? TrackStatus::Confirmed
                             : TrackStatus::Tentative;
      new_track.history.reserve(kHistoryLength);
      new_track.history.emplace_back(new_track.state.mean(0),
                                     new_track.state.mean(1));
      new_tracks.push_back(std::move(new_track));
    }
  }

  for (Track& track : new_tracks) tracks_.push_back(std::move(track));

  marked_for_deletion.resize(tracks_.size(), false);
  std::size_t erase_cursor = 0;
  std::erase_if(tracks_, [&](const Track&) {
    return marked_for_deletion[erase_cursor++];
  });
}

// Hands back every track exactly as the last step call left it,
// tentative ones included. The determinism and allocation tests read
// the tracker through this, since they care about the raw life cycle
// rather than what the viewer or the metrics choose to show.
const std::vector<Track>& Tracker::tracks() const {
  return tracks_;
}

// Hands back the tracks a caller outside the tracker is allowed to
// see: only the ones confirmed enough to report, carried forward in
// time to the moment asked for. This is what the viewer draws and
// what the metrics score, so a track that has not yet earned enough
// hits, or one that only just started coasting on a single miss,
// never reaches either of them.
std::vector<Track> Tracker::confirmed_tracks_at(
    std::int64_t query_time_micros) const {
  std::vector<Track> result;
  for (const Track& track : tracks_) {
    if (track.status != TrackStatus::Confirmed &&
        track.status != TrackStatus::Coasting)
      continue;
    if (track.hits < settings_.hits_to_confirm) continue;
    Track copy = track;
    if (query_time_micros > copy.state_time_micros) {
      predict(copy.state,
              static_cast<double>(query_time_micros -
                                  copy.state_time_micros) /
                  1e6,
              settings_.noise);
      copy.state_time_micros = query_time_micros;
    }
    result.push_back(std::move(copy));
  }
  return result;
}
