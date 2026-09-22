// Steps a tracker through a staged segment's frames and records
// what it confirms; used by the CLI and the determinism test.

#pragma once
#include <cstdint>
#include <vector>
#include "log.hpp"
#include "perturb.hpp"
#include "tracker.hpp"

// Seed and settings a Replay run uses for perturbation and for
// the tracker underneath it.
struct ReplaySettings {
  std::uint64_t seed = 1;
  PerturbationSettings perturbation;
  TrackerSettings tracker;
};

// Per-step wall-clock timings in milliseconds, plus a count of
// steps slower than the segment's frame period.
struct TimingStats {
  std::vector<double> step_milliseconds;
  std::uint64_t overruns = 0;
  // Step time in milliseconds at the given fraction, e.g. 0.5 for p50.
  double percentile(double fraction) const;
};

// Steps one tracker instance through one segment; not shared
// across segments or threads.
class Replay {
 public:
  // Builds a Replay over segment (which must outlive it) using settings.
  Replay(const SegmentLog& segment, ReplaySettings settings);
  // Runs the segment: perturbs, queues by arrival time, and steps the tracker.
  void run();
  // Step timings recorded by the most recent run() call.
  const TimingStats& timing() const;
  // Confirmed tracks from the most recent run() call, one entry per frame.
  const std::vector<std::vector<Track>>& confirmed_tracks_per_frame() const;

 private:
  const SegmentLog& segment_;
  Tracker tracker_;
  Perturbation perturbation_;
  TimingStats timing_;
  std::vector<std::vector<Track>> confirmed_tracks_per_frame_;
};
