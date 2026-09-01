#pragma once
#include <cstdint>
#include <vector>
#include "log.hpp"
#include "perturb.hpp"
#include "tracker.hpp"

struct ReplaySettings {
  std::uint64_t seed = 1;
  PerturbationSettings perturbation;
  TrackerSettings tracker;
};

struct TimingStats {
  std::vector<double> step_milliseconds;
  std::uint64_t overruns = 0;
  double percentile(double fraction) const;
};

class Replay {
 public:
  Replay(const SegmentLog& segment, ReplaySettings settings);
  void run();
  const TimingStats& timing() const;
  const std::vector<std::vector<Track>>& confirmed_tracks_per_frame()
      const;

 private:
  const SegmentLog& segment_;
  Tracker tracker_;
  Perturbation perturbation_;
  TimingStats timing_;
  std::vector<std::vector<Track>> confirmed_tracks_per_frame_;
};
