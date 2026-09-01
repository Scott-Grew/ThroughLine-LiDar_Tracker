#include "perturb.hpp"

// This file stands in for everything that makes a real sensor worse
// than a perfect one. It sits between the ground truth and the
// tracker: replay.cpp reads a frame's detections, passes them through
// here, and only the result is ever handed to the tracker. Three
// faults are modelled - a detection can be dropped entirely, its box
// can be reported in the wrong place, and the whole detection can
// arrive late. Turning any of these up is how a run answers "how well
// does this tracker hold up on a worse sensor" without touching the
// tracker itself.

Perturbation::Perturbation(PerturbationSettings settings,
                           std::uint64_t seed)
    : settings_(settings), generator_(seed) {}

// Runs every detection in a frame through the drop and displacement
// faults, in a fixed order so that two runs with the same seed make
// the same decisions for the same detection every time. Arrival time
// is available_time's job below.
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

// When a detection captured at this time actually reaches the
// tracker, once the sensor's own processing and transmission delay is
// accounted for. The tracker never sees a detection before this time,
// which is what lets a frame's late detections turn up attached to a
// later frame.
std::int64_t Perturbation::available_time(
    std::int64_t capture_time_micros) const {
  return capture_time_micros + settings_.latency_micros;
}
