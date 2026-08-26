#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "types.hpp"

struct ClassMetrics {
  std::uint64_t matches = 0, misses = 0, false_positives = 0, id_switches = 0, fragmentations = 0, ground_truth_total = 0;
  double iou_sum = 0.0;
  double mota() const;
  double motp() const;
};

class TrackingMetrics {
 public:
  void update(const std::vector<Track>& tracks, const std::vector<GroundTruthBox>& ground_truth, const Eigen::Matrix4d& vehicle_to_world);
  const std::unordered_map<ObjectClass, ClassMetrics>& per_class() const;
 private:
  std::unordered_map<ObjectClass, ClassMetrics> per_class_;
  std::unordered_map<std::uint64_t, std::uint64_t> last_matched_track_for_object_;
  std::unordered_map<std::uint64_t, bool> matched_last_frame_;
  std::unordered_map<std::uint64_t, bool> ever_matched_;
};
