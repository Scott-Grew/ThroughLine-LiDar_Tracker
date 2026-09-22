// Declares the tracker and the knobs that govern its life cycle.

#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "assign.hpp"
#include "filter.hpp"
#include "types.hpp"

// Tuning knobs for track confirmation, deletion, the assignment
// gate, and the filter's own process and measurement noise.
struct TrackerSettings {
  // Matched frames before a track is reported. Chosen.
  std::uint32_t hits_to_confirm = 3;
  // Unmatched frames before a confirmed track is dropped. Chosen.
  std::uint32_t misses_to_delete = 5;
  // Max squared Mahalanobis distance for a match: chi-squared, 3
  // degrees of freedom, 0.99 quantile. Refuses 1% of true matches.
  double gate_chi_squared = 11.34;
  // Track slots reserved up front; the list reallocates only if
  // more tracks than this are alive at once. Chosen.
  std::size_t reserved_tracks = 512;
  AssignmentMethod assignment = AssignmentMethod::Hungarian;
  FilterNoise noise;
};

// Owns the track list, held in the world frame, and decides frame
// by frame which detection belongs to which track. One thread only.
class Tracker {
 public:
  // Builds a tracker sized by settings, with its track list pre-reserved.
  explicit Tracker(TrackerSettings settings);
  // Runs one frame's predict/match/update/create/delete life cycle.
  void step(std::int64_t capture_time_micros,
            const std::vector<Detection>& detections,
            const Eigen::Matrix4d& vehicle_to_world);
  // All tracks as the last step() call left them, tentative ones included.
  const std::vector<Track>& tracks() const;
  // Confirmed and coasting tracks predicted forward to query_time_micros.
  std::vector<Track> confirmed_tracks_at(std::int64_t query_time_micros) const;

 private:
  void predict_all_to(std::int64_t capture_time_micros);
  static Box to_world(const Box& box, const Eigen::Matrix4d& vehicle_to_world);
  static std::vector<Box> boxes_in_world(
      const std::vector<Detection>& detections,
      const Eigen::Matrix4d& vehicle_to_world);
  std::vector<int> track_indices_of(ObjectClass object_class) const;
  static std::vector<int> detection_indices_of(
      const std::vector<Detection>& detections, ObjectClass object_class);
  void fill_cost(Eigen::MatrixXd& cost, const std::vector<int>& track_indices,
                 const std::vector<int>& detection_indices,
                 const std::vector<Box>& world_boxes) const;
  TrackStatus status_for_hits(std::uint32_t hits) const;
  void update_matched(Track& track, const Box& world_box);
  bool register_miss(Track& track);
  Track start_track(ObjectClass object_class, const Box& world_box,
                    std::int64_t capture_time_micros);
  void erase_marked(std::vector<bool>& marked_for_deletion);

  TrackerSettings settings_;
  std::vector<Track> tracks_;
  std::uint64_t next_track_id_ = 1;
  std::unordered_map<ObjectClass, Eigen::MatrixXd> cost_by_class_;
};
