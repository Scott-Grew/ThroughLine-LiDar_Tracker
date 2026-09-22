// Fault injector standing between ground truth and the tracker;
// replay.cpp passes each frame's detections through apply() first.

#include "perturb.hpp"

Perturbation::Perturbation(PerturbationSettings settings, std::uint64_t seed)
    : settings_(settings), generator_(seed) {}

std::vector<Detection> Perturbation::apply(
    const std::vector<Detection>& detections) {
  std::uniform_real_distribution<double> dropout_draw(0.0, 1.0);
  std::vector<Detection> surviving_detections;
  surviving_detections.reserve(detections.size());
  for (const Detection& detection : detections) {
    if (dropout_draw(generator_) < settings_.dropout_probability) continue;
    Detection perturbed_detection = detection;
    if (settings_.position_noise_metres > 0.0) {
      std::normal_distribution<double> position_noise(
          0.0, settings_.position_noise_metres);
      perturbed_detection.box.center_x += position_noise(generator_);
      perturbed_detection.box.center_y += position_noise(generator_);
    }
    surviving_detections.push_back(perturbed_detection);
  }
  return surviving_detections;
}

std::int64_t Perturbation::available_time(
    std::int64_t capture_time_micros) const {
  return capture_time_micros + settings_.latency_micros;
}
