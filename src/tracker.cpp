// Frame-by-frame track life cycle: predict, match, update, create
// and delete tracks. Replay reads the confirmed tracks this owns.

#include "tracker.hpp"

#include <cmath>

namespace {

// Trajectory points kept per track, oldest dropped first.
constexpr std::size_t kHistoryLength = 30;

// Elapsed time between two capture timestamps, in seconds.
double seconds_between(std::int64_t from_micros, std::int64_t to_micros) {
  return static_cast<double>(to_micros - from_micros) /
         static_cast<double>(kMicrosPerSecond);
}

}  // namespace

// Reserves reserved_tracks slots so step() does not reallocate
// until more tracks than that are alive at once.
Tracker::Tracker(TrackerSettings settings) : settings_(settings) {
  tracks_.reserve(settings_.reserved_tracks);
}

// Converts a box from the vehicle frame into the fixed world frame
// using the frame's vehicle pose; matching happens in world.
Box Tracker::to_world(const Box& box, const Eigen::Matrix4d& vehicle_to_world) {
  const Eigen::Vector4d local_position(box.center_x, box.center_y, box.center_z,
                                       1.0);
  const Eigen::Vector4d world_position = vehicle_to_world * local_position;
  Box world_box = box;
  world_box.center_x = world_position(0);
  world_box.center_y = world_position(1);
  world_box.center_z = world_position(2);
  world_box.yaw = wrap_angle(
      box.yaw + std::atan2(vehicle_to_world(1, 0), vehicle_to_world(0, 0)));
  return world_box;
}

// Advances each track's filter to this frame's capture time. Each
// track's state_time_micros stops it being predicted twice per frame.
void Tracker::predict_all_to(std::int64_t capture_time_micros) {
  for (Track& track : tracks_) {
    if (track.state_time_micros >= capture_time_micros) continue;
    const double dt_seconds =
        seconds_between(track.state_time_micros, capture_time_micros);
    predict(track.state, dt_seconds, settings_.noise);
    track.state_time_micros = capture_time_micros;
  }
}

// Converts each detection's box into the world frame, keeping
// detection order so index lists built from it still line up.
std::vector<Box> Tracker::boxes_in_world(
    const std::vector<Detection>& detections,
    const Eigen::Matrix4d& vehicle_to_world) {
  std::vector<Box> world_boxes;
  world_boxes.reserve(detections.size());
  for (const Detection& detection : detections)
    world_boxes.push_back(to_world(detection.box, vehicle_to_world));
  return world_boxes;
}

// Indices into tracks_ belonging to one object class, in track order.
std::vector<int> Tracker::track_indices_of(ObjectClass object_class) const {
  std::vector<int> track_indices;
  for (std::size_t index = 0; index < tracks_.size(); ++index)
    if (tracks_[index].object_class == object_class)
      track_indices.push_back(static_cast<int>(index));
  return track_indices;
}

// Indices into detections belonging to one object class, in detection order.
std::vector<int> Tracker::detection_indices_of(
    const std::vector<Detection>& detections, ObjectClass object_class) {
  std::vector<int> detection_indices;
  for (std::size_t index = 0; index < detections.size(); ++index)
    if (detections[index].object_class == object_class)
      detection_indices.push_back(static_cast<int>(index));
  return detection_indices;
}

// Fills one class's cost matrix with squared Mahalanobis distance
// between each candidate track and each candidate detection.
void Tracker::fill_cost(Eigen::MatrixXd& cost,
                        const std::vector<int>& track_indices,
                        const std::vector<int>& detection_indices,
                        const std::vector<Box>& world_boxes) const {
  cost.resize(track_indices.size(), detection_indices.size());
  for (int row = 0; row < static_cast<int>(track_indices.size()); ++row)
    for (int column = 0; column < static_cast<int>(detection_indices.size());
         ++column)
      cost(row, column) = mahalanobis_squared(
          tracks_[track_indices[row]].state,
          world_boxes[detection_indices[column]], settings_.noise);
}

// Confirmation threshold shared by matched updates and track creation.
TrackStatus Tracker::status_for_hits(std::uint32_t hits) const {
  return hits >= settings_.hits_to_confirm ? TrackStatus::Confirmed
                                           : TrackStatus::Tentative;
}

// Applies one matched detection to its track's filter and history.
void Tracker::update_matched(Track& track, const Box& world_box) {
  update(track.state, world_box, settings_.noise);
  track.hits += 1;
  track.consecutive_misses = 0;
  track.status = status_for_hits(track.hits);
  track.history.emplace_back(track.state.mean(kPositionXIndex),
                             track.state.mean(kPositionYIndex));
  if (track.history.size() > kHistoryLength)
    track.history.erase(track.history.begin());
}

// Advances one unmatched track's miss count and reports whether it
// must be erased this step.
bool Tracker::register_miss(Track& track) {
  track.consecutive_misses += 1;
  if (track.status == TrackStatus::Tentative) return true;
  track.status = TrackStatus::Coasting;
  return track.consecutive_misses >= settings_.misses_to_delete;
}

// Builds a track for an unmatched detection at this frame's
// capture time, consuming the next track id.
Track Tracker::start_track(ObjectClass object_class, const Box& world_box,
                           std::int64_t capture_time_micros) {
  Track new_track;
  new_track.track_id = next_track_id_++;
  new_track.object_class = object_class;
  new_track.state = initial_state(world_box, settings_.noise);
  new_track.state_time_micros = capture_time_micros;
  new_track.hits = 1;
  new_track.consecutive_misses = 0;
  new_track.status = status_for_hits(new_track.hits);
  new_track.history.reserve(kHistoryLength);
  new_track.history.emplace_back(new_track.state.mean(kPositionXIndex),
                                 new_track.state.mean(kPositionYIndex));
  return new_track;
}

// Removes each track marked for deletion from tracks_ in place.
void Tracker::erase_marked(std::vector<bool>& marked_for_deletion) {
  // marked_for_deletion mirrors tracks_ by index; erase_cursor keeps
  // the two in step as erase_if visits tracks_ in order.
  marked_for_deletion.resize(tracks_.size(), false);
  std::size_t erase_cursor = 0;
  std::erase_if(tracks_, [&](const Track&) {
    return marked_for_deletion[erase_cursor++];
  });
}

// Runs one frame's full life cycle: predict, assign detections to
// tracks per object class, update matches, create and delete tracks.
void Tracker::step(std::int64_t capture_time_micros,
                   const std::vector<Detection>& detections,
                   const Eigen::Matrix4d& vehicle_to_world) {
  predict_all_to(capture_time_micros);

  const std::vector<Box> world_boxes =
      boxes_in_world(detections, vehicle_to_world);

  std::vector<bool> marked_for_deletion(tracks_.size(), false);
  std::vector<Track> new_tracks;

  for (ObjectClass object_class :
       {ObjectClass::Vehicle, ObjectClass::Pedestrian, ObjectClass::Cyclist}) {
    const std::vector<int> track_indices = track_indices_of(object_class);
    const std::vector<int> detection_indices =
        detection_indices_of(detections, object_class);

    Eigen::MatrixXd& cost = cost_by_class_[object_class];
    fill_cost(cost, track_indices, detection_indices, world_boxes);

    const Assignment assignment =
        assign(cost, settings_.gate_chi_squared, settings_.assignment);

    for (const auto& [row, column] : assignment.pairs)
      update_matched(tracks_[track_indices[row]],
                     world_boxes[detection_indices[column]]);

    for (int unmatched_row : assignment.unmatched_rows) {
      Track& track = tracks_[track_indices[unmatched_row]];
      if (register_miss(track))
        marked_for_deletion[track_indices[unmatched_row]] = true;
    }

    for (int unmatched_column : assignment.unmatched_columns) {
      const Box& world_box = world_boxes[detection_indices[unmatched_column]];
      new_tracks.push_back(
          start_track(object_class, world_box, capture_time_micros));
    }
  }

  for (Track& track : new_tracks) tracks_.push_back(std::move(track));

  erase_marked(marked_for_deletion);
}

// All tracks exactly as the last step() call left them, tentative
// tracks included.
const std::vector<Track>& Tracker::tracks() const {
  return tracks_;
}

// Copies of the confirmed and coasting tracks, predicted forward to
// the given time; this is what gets exported and recorded.
std::vector<Track> Tracker::confirmed_tracks_at(
    std::int64_t query_time_micros) const {
  std::vector<Track> confirmed_tracks;
  for (const Track& track : tracks_) {
    if (track.status != TrackStatus::Confirmed &&
        track.status != TrackStatus::Coasting)
      continue;
    if (track.hits < settings_.hits_to_confirm) continue;
    Track predicted_track = track;
    if (query_time_micros > predicted_track.state_time_micros) {
      predict(
          predicted_track.state,
          seconds_between(predicted_track.state_time_micros, query_time_micros),
          settings_.noise);
      predicted_track.state_time_micros = query_time_micros;
    }
    confirmed_tracks.push_back(std::move(predicted_track));
  }
  return confirmed_tracks;
}
