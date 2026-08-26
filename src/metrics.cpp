#include "metrics.hpp"

namespace {
[[maybe_unused]] constexpr double kIouThresholdVehicle = 0.7;
[[maybe_unused]] constexpr double kIouThresholdPedestrian = 0.5;
[[maybe_unused]] constexpr double kIouThresholdCyclist = 0.5;
}

double ClassMetrics::mota() const {
  return 0.0;
}

double ClassMetrics::motp() const {
  return 0.0;
}

void TrackingMetrics::update(const std::vector<Track>& tracks, const std::vector<GroundTruthBox>& ground_truth, const Eigen::Matrix4d& vehicle_to_world) {
  (void)tracks;
  (void)ground_truth;
  (void)vehicle_to_world;
}

const std::unordered_map<ObjectClass, ClassMetrics>& TrackingMetrics::per_class() const {
  return per_class_;
}
