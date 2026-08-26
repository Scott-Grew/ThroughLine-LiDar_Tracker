#include "tracker.hpp"

Tracker::Tracker(TrackerSettings settings) : settings_(settings) {}

void Tracker::step(std::int64_t capture_time_micros, const std::vector<Detection>& detections, const Eigen::Matrix4d& vehicle_to_world) {
  (void)capture_time_micros;
  (void)detections;
  (void)vehicle_to_world;
}

const std::vector<Track>& Tracker::tracks() const {
  return tracks_;
}

std::vector<Track> Tracker::confirmed_tracks_at(std::int64_t query_time_micros) const {
  (void)query_time_micros;
  return {};
}
