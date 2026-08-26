#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "assign.hpp"
#include "filter.hpp"
#include "types.hpp"

struct TrackerSettings {
  std::uint32_t hits_to_confirm = 3;
  std::uint32_t misses_to_delete = 5;
  double gate_chi_squared = 11.34;
  std::size_t reserved_tracks = 512;
  AssignmentMethod assignment = AssignmentMethod::Hungarian;
  FilterNoise noise;
};

class Tracker {
 public:
  explicit Tracker(TrackerSettings settings);
  void step(std::int64_t capture_time_micros, const std::vector<Detection>& detections, const Eigen::Matrix4d& vehicle_to_world);
  const std::vector<Track>& tracks() const;
  std::vector<Track> confirmed_tracks_at(std::int64_t query_time_micros) const;
 private:
  void predict_all_to(std::int64_t capture_time_micros);
  static Box to_world(const Box& box, const Eigen::Matrix4d& vehicle_to_world);

  TrackerSettings settings_;
  std::vector<Track> tracks_;
  std::uint64_t next_track_id_ = 1;
  std::unordered_map<ObjectClass, Eigen::MatrixXd> cost_by_class_;
};
