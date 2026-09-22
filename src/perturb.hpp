// Declares the sensor-fault model applied between ground truth and
// the tracker: dropout, position noise, and arrival latency.

#pragma once
#include <cstdint>
#include <random>
#include <vector>
#include "types.hpp"

// Fault rates for one run: drop probability, position noise sigma in
// metres, and arrival delay in microseconds.
struct PerturbationSettings {
  double dropout_probability = 0.0;
  double position_noise_metres = 0.0;
  std::int64_t latency_micros = 0;
};

// Applies dropout and position noise to a frame's detections, and
// reports when a detection captured earlier actually arrives.
class Perturbation {
 public:
  // Builds a Perturbation seeded so its faults are reproducible.
  Perturbation(PerturbationSettings settings, std::uint64_t seed);
  // Detections that survive dropout, with position noise applied.
  std::vector<Detection> apply(const std::vector<Detection>& detections);
  // Time the detection captured at capture_time_micros reaches the tracker.
  std::int64_t available_time(std::int64_t capture_time_micros) const;

 private:
  PerturbationSettings settings_;
  std::mt19937_64 generator_;
};
