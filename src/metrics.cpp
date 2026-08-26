#include "metrics.hpp"

#include <cmath>
#include "assign.hpp"
#include "filter.hpp"
#include "iou.hpp"

// This file scores the tracker against ground truth, using the same CLEAR metrics Waymo's own
// evaluator reports: MOTA, which counts up every mistake a tracker can make per ground truth
// object, and MOTP, which measures how tightly the boxes that were matched actually line up.
// Replay.cpp calls update once per frame with whatever the tracker just confirmed; main.cpp prints
// the running totals at the end of a run, and the viewer shows them live. This is a training
// monitor, not the reported score - the number that counts still comes from Waymo's own evaluator
// running in the container, on the CLEAR rules it actually judges by.

namespace {

constexpr double kIouThresholdVehicle = 0.7;
constexpr double kIouThresholdPedestrian = 0.5;
constexpr double kIouThresholdCyclist = 0.5;

// Waymo scores each object class against its own overlap bar - a pedestrian counts as matched at
// a much looser overlap than a vehicle does, because a pedestrian's footprint is tiny to begin
// with.
double iou_threshold_for(ObjectClass object_class) {
  if (object_class == ObjectClass::Vehicle) return kIouThresholdVehicle;
  if (object_class == ObjectClass::Pedestrian) return kIouThresholdPedestrian;
  return kIouThresholdCyclist;
}

// The same sensor-frame-to-world transform the tracker itself uses, kept private to this file so
// that scoring never depends on a helper the tracker exposes just for its own sake.
Box to_world(const Box& box, const Eigen::Matrix4d& vehicle_to_world) {
  const Eigen::Vector4d local_position(box.center_x, box.center_y, box.center_z, 1.0);
  const Eigen::Vector4d world_position = vehicle_to_world * local_position;
  Box world_box = box;
  world_box.center_x = world_position(0);
  world_box.center_y = world_position(1);
  world_box.center_z = world_position(2);
  world_box.yaw = wrap_angle(box.yaw + std::atan2(vehicle_to_world(1, 0), vehicle_to_world(0, 0)));
  return world_box;
}

// The box a track currently claims to occupy, read out of the filter's five numbers plus the size
// it has been smoothing. This is the only place metrics ever looks inside a track's state.
Box track_box(const Track& track) {
  return Box{track.state.mean(0), track.state.mean(1), track.state.center_z,
             track.state.length, track.state.width, track.state.height, track.state.mean(2)};
}

// One ground truth object paired with the track that was decided to be the same object this
// frame, and how well their boxes actually overlapped.
struct MatchedPair {
  std::size_t ground_truth_index;
  std::size_t track_index;
  double intersection_over_union;
};

}

// Mistakes per ground truth object that existed: a track that never showed up counts the same as
// a track that showed up but was wrong. A class with no ground truth this run has nothing to be
// wrong about, so it scores a perfect one rather than dividing by zero.
double ClassMetrics::mota() const {
  if (ground_truth_total == 0) return 0.0;
  const double mistakes = static_cast<double>(misses + false_positives + id_switches);
  return 1.0 - mistakes / static_cast<double>(ground_truth_total);
}

// How tight the matched boxes were on average, as a mean overlap rather than a mean error, so
// higher is still better. A class that has never matched anything has no precision to report.
double ClassMetrics::motp() const {
  if (matches == 0) return 0.0;
  return iou_sum / static_cast<double>(matches);
}

// Scores one frame: for each object class, decides which confirmed track goes with which ground
// truth box, and folds the result into that class's running totals. A ground truth object keeps
// its previous track across a frame whenever that exact track is still around and still close
// enough, so a match is never thrown away and reassigned by the optimal solver just because
// another track briefly became a fraction closer - that is what an id switch is supposed to
// penalise, not something the scorer should manufacture on its own. Everything left over after
// that continuity check is matched optimally by overlap.
void TrackingMetrics::update(const std::vector<Track>& tracks, const std::vector<GroundTruthBox>& ground_truth, const Eigen::Matrix4d& vehicle_to_world) {
  for (ObjectClass object_class : {ObjectClass::Vehicle, ObjectClass::Pedestrian, ObjectClass::Cyclist}) {
    const double iou_threshold = iou_threshold_for(object_class);
    ClassMetrics& class_metrics = per_class_[object_class];

    std::vector<GroundTruthBox> eligible_ground_truth;
    for (const GroundTruthBox& truth : ground_truth) {
      if (truth.object_class != object_class || truth.lidar_points_in_box <= 0) continue;
      GroundTruthBox world_truth = truth;
      world_truth.box = to_world(truth.box, vehicle_to_world);
      eligible_ground_truth.push_back(world_truth);
    }
    class_metrics.ground_truth_total += eligible_ground_truth.size();

    std::vector<const Track*> candidate_tracks;
    for (const Track& track : tracks)
      if (track.object_class == object_class) candidate_tracks.push_back(&track);

    std::vector<bool> ground_truth_claimed(eligible_ground_truth.size(), false);
    std::vector<bool> track_claimed(candidate_tracks.size(), false);
    std::vector<MatchedPair> matched_pairs;

    for (std::size_t ground_truth_index = 0; ground_truth_index < eligible_ground_truth.size(); ++ground_truth_index) {
      const GroundTruthBox& truth = eligible_ground_truth[ground_truth_index];
      const auto continuity = last_matched_track_for_object_.find(truth.object_id);
      if (continuity == last_matched_track_for_object_.end()) continue;
      for (std::size_t track_index = 0; track_index < candidate_tracks.size(); ++track_index) {
        if (track_claimed[track_index] || candidate_tracks[track_index]->track_id != continuity->second) continue;
        const double iou = intersection_over_union_3d(track_box(*candidate_tracks[track_index]), truth.box);
        if (iou >= iou_threshold) {
          matched_pairs.push_back({ground_truth_index, track_index, iou});
          ground_truth_claimed[ground_truth_index] = true;
          track_claimed[track_index] = true;
        }
        break;
      }
    }

    std::vector<std::size_t> remaining_ground_truth_indices;
    for (std::size_t ground_truth_index = 0; ground_truth_index < eligible_ground_truth.size(); ++ground_truth_index)
      if (!ground_truth_claimed[ground_truth_index]) remaining_ground_truth_indices.push_back(ground_truth_index);

    std::vector<std::size_t> remaining_track_indices;
    for (std::size_t track_index = 0; track_index < candidate_tracks.size(); ++track_index)
      if (!track_claimed[track_index]) remaining_track_indices.push_back(track_index);

    Eigen::MatrixXd cost(remaining_track_indices.size(), remaining_ground_truth_indices.size());
    for (int row = 0; row < static_cast<int>(remaining_track_indices.size()); ++row)
      for (int column = 0; column < static_cast<int>(remaining_ground_truth_indices.size()); ++column)
        cost(row, column) = 1.0 - intersection_over_union_3d(track_box(*candidate_tracks[remaining_track_indices[row]]),
                                                               eligible_ground_truth[remaining_ground_truth_indices[column]].box);

    const Assignment assignment = assign(cost, 1.0 - iou_threshold, AssignmentMethod::Hungarian);
    for (const auto& [row, column] : assignment.pairs) {
      const std::size_t track_index = remaining_track_indices[row];
      const std::size_t ground_truth_index = remaining_ground_truth_indices[column];
      matched_pairs.push_back({ground_truth_index, track_index, 1.0 - cost(row, column)});
      ground_truth_claimed[ground_truth_index] = true;
      track_claimed[track_index] = true;
    }

    for (const MatchedPair& pair : matched_pairs) {
      const GroundTruthBox& truth = eligible_ground_truth[pair.ground_truth_index];
      const Track& track = *candidate_tracks[pair.track_index];
      class_metrics.matches += 1;
      class_metrics.iou_sum += pair.intersection_over_union;
      const auto previous_match = last_matched_track_for_object_.find(truth.object_id);
      if (previous_match != last_matched_track_for_object_.end() && previous_match->second != track.track_id) class_metrics.id_switches += 1;
      last_matched_track_for_object_[truth.object_id] = track.track_id;
    }

    for (std::size_t ground_truth_index = 0; ground_truth_index < eligible_ground_truth.size(); ++ground_truth_index) {
      if (ground_truth_claimed[ground_truth_index]) continue;
      class_metrics.misses += 1;
    }

    for (std::size_t track_index = 0; track_index < candidate_tracks.size(); ++track_index)
      if (!track_claimed[track_index]) class_metrics.false_positives += 1;
  }
}

// Hands back the running totals for every class seen so far, which is what main.cpp prints and
// the viewer displays live.
const std::unordered_map<ObjectClass, ClassMetrics>& TrackingMetrics::per_class() const {
  return per_class_;
}
