#include "perturb.hpp"

// Fault injector standing between ground truth and the tracker;
// replay.cpp passes every frame's detections through apply() first.

// Seeds the fault generator so a run is reproducible from the seed.
Perturbation::Perturbation(PerturbationSettings settings,
                           std::uint64_t seed)
    : settings_(settings), generator_(seed) {}

// Applies dropout and position noise to each detection in a fixed
// order, so the same seed reproduces the same faults every run.
std::vector<Detection> Perturbation::apply(
    const std::vector<Detection>& detections) {
  std::uniform_real_distribution<double> dropout_draw(0.0, 1.0);
  std::vector<Detection> kept;
  kept.reserve(detections.size());
  for (const Detection& detection : detections) {
    if (dropout_draw(generator_) < settings_.dropout_probability)
      continue;
    Detection perturbed = detection;
    if (settings_.position_noise_metres > 0.0) {
      std::normal_distribution<double> position_noise(
          0.0, settings_.position_noise_metres);
      perturbed.box.center_x += position_noise(generator_);
      perturbed.box.center_y += position_noise(generator_);
    }
    kept.push_back(perturbed);
  }
  return kept;
}

// The time a captured detection actually reaches the tracker, after
// the modelled latency.
std::int64_t Perturbation::available_time(
    std::int64_t capture_time_micros) const {
  return capture_time_micros + settings_.latency_micros;
}
