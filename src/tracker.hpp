// Declares the tracker and the knobs that govern its life cycle. main.cpp
// steps it one frame at a time and reads the confirmed tracks back.

#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "filter.hpp"
#include "types.hpp"

// Tuning knobs for track confirmation, deletion, the assignment
// gate, and the filter's own process and measurement noise.
struct TrackerSettings {
  // Matched frames before a track is reported. Chosen.
  std::uint32_t hits_to_confirm = 3;
  // Unmatched frames before a confirmed track is dropped. Chosen.
  std::uint32_t misses_to_delete = 5;
  // Max squared Mahalanobis distance for a match, the chi-squared 0.99
  // quantile at 3 degrees of freedom. A track left unmatched coasts.
  double gate_chi_squared = 11.34;
  FilterNoise noise;
};

// The matched row/column pairs from one assign() call, plus the rows
// and columns left unmatched.
struct Assignment {
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_columns;
};

// Minimum-cost matching of rows to columns over the pairs whose cost is
// within the gate, with as many pairs as possible.
Assignment assign(const Eigen::MatrixXd& cost, double gate);

// Owns the track list, held in the world frame, and decides frame by
// frame which detection belongs to which track. One thread only.
class Tracker {
 public:
  explicit Tracker(TrackerSettings settings);
  void step(const Frame& frame);
  std::vector<Track> confirmed_at(std::int64_t time_micros) const;

 private:
  Eigen::MatrixXd costs(const std::vector<Detection>& detections) const;
  Track new_track(const Detection& detection, std::int64_t time_micros);
  void update_matched(Track& track, const Box& box) const;
  TrackStatus next_status(const Track& track) const;

  TrackerSettings settings_;
  MotionFilter filter_;
  std::vector<Track> tracks_;
  std::uint64_t next_track_id_ = 1;
};
