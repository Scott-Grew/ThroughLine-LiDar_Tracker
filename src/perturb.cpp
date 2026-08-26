#include "perturb.hpp"

Perturbation::Perturbation(PerturbationSettings settings, std::uint64_t seed)
    : settings_(settings), generator_(seed) {}

std::vector<Detection> Perturbation::apply(const std::vector<Detection>& detections) {
  (void)detections;
  return {};
}

std::int64_t Perturbation::available_time(std::int64_t capture_time_micros) const {
  (void)capture_time_micros;
  return 0;
}
