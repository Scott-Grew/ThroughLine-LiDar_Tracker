#pragma once
#include <cstdint>
#include <random>
#include <vector>
#include "types.hpp"

// Declares the sensor-fault model applied between ground truth and
// the tracker: dropout, position noise, and arrival latency.

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
  Perturbation(PerturbationSettings settings, std::uint64_t seed);
  std::vector<Detection> apply(
      const std::vector<Detection>& detections);
  std::int64_t available_time(std::int64_t capture_time_micros) const;

 private:
  PerturbationSettings settings_;
  std::mt19937_64 generator_;
};
