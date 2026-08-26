#include "replay.hpp"

double TimingStats::percentile(double fraction) const {
  (void)fraction;
  return 0.0;
}

Snapshot& SnapshotExchange::writable() {
  return buffers_[writing_];
}

void SnapshotExchange::publish() {
  published_.store(writing_);
  writing_ = (writing_ + 1) % static_cast<int>(buffers_.size());
}

const Snapshot* SnapshotExchange::latest() const {
  int published_index = published_.load();
  if (published_index < 0) {
    return nullptr;
  }
  return &buffers_[published_index];
}

Replay::Replay(const SegmentLog& segment, ReplaySettings settings, const Predictor& predictor)
    : segment_(segment),
      settings_(settings),
      predictor_(predictor),
      tracker_(settings.tracker),
      perturbation_(settings.perturbation, settings.seed) {}

void Replay::run(SnapshotExchange& exchange, LiveControls& controls) {
  (void)exchange;
  (void)controls;
}

const TrackingMetrics& Replay::metrics() const {
  return metrics_;
}

const TimingStats& Replay::timing() const {
  return timing_;
}

const std::vector<std::vector<Track>>& Replay::confirmed_tracks_per_frame() const {
  return confirmed_tracks_per_frame_;
}
