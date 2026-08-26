#pragma once
#include <cstdint>
#include <random>
#include "types.hpp"

struct PerturbationSettings {
  double dropout_probability = 0.0;
  double position_noise_metres = 0.0;
  std::int64_t latency_micros = 0;
};

class Perturbation {
 public:
  Perturbation(PerturbationSettings settings, std::uint64_t seed);
  void set(PerturbationSettings settings);
  std::vector<Detection> apply(const std::vector<Detection>& detections);
  std::int64_t available_time(std::int64_t capture_time_micros) const;
 private:
  PerturbationSettings settings_;
  std::mt19937_64 generator_;
};
