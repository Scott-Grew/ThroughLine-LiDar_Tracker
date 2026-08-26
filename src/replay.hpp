#pragma once
#include <atomic>
#include <array>
#include <cstdint>
#include <vector>
#include "log.hpp"
#include "metrics.hpp"
#include "perturb.hpp"
#include "predict.hpp"
#include "tracker.hpp"

enum class DetectionSource { GroundTruth, Detector };

struct ReplaySettings {
  DetectionSource source = DetectionSource::GroundTruth;
  double rate = 1.0;
  bool headless = false;
  std::uint64_t seed = 1;
  double prediction_horizon_seconds = 3.0;
  double prediction_step_seconds = 0.1;
  PerturbationSettings perturbation;
  TrackerSettings tracker;
};

struct TimingStats {
  std::vector<double> step_milliseconds;
  std::uint64_t overruns = 0;
  double percentile(double fraction) const;
};

struct Snapshot {
  std::size_t frame_index = 0;
  const Frame* frame = nullptr;
  std::vector<Track> tracks;
  std::vector<PredictedPath> predictions;
  std::unordered_map<ObjectClass, ClassMetrics> metrics;
  double step_p50_milliseconds = 0.0, step_p99_milliseconds = 0.0;
  std::uint64_t overruns = 0;
};

class SnapshotExchange {
 public:
  Snapshot& writable();
  void publish();
  const Snapshot* acquire();
 private:
  static constexpr int kFreshFlag = 4;
  std::array<Snapshot, 3> buffers_;
  int writing_ = 0;
  int reading_ = 1;
  std::atomic<int> published_{2};
};

struct LiveControls {
  std::atomic<double> dropout_probability{0.0};
  std::atomic<double> position_noise_metres{0.0};
  std::atomic<int64_t> latency_micros{0};
  std::atomic<bool> paused{false};
  std::atomic<bool> quit{false};
};

class Replay {
 public:
  Replay(const SegmentLog& segment, ReplaySettings settings, const Predictor& predictor);
  void run(SnapshotExchange& exchange, LiveControls& controls);
  const TrackingMetrics& metrics() const;
  const TimingStats& timing() const;
  const std::vector<std::vector<Track>>& confirmed_tracks_per_frame() const;
 private:
  const SegmentLog& segment_;
  ReplaySettings settings_;
  const Predictor& predictor_;
  Tracker tracker_;
  Perturbation perturbation_;
  TrackingMetrics metrics_;
  TimingStats timing_;
  std::vector<std::vector<Track>> confirmed_tracks_per_frame_;
};
