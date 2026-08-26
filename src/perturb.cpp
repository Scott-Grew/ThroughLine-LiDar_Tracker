#include "perturb.hpp"

#include "filter.hpp"

// This file stands in for everything that makes a real sensor worse than a perfect one. It sits
// between the ground truth or a detector and the tracker: replay.cpp reads a frame's detections,
// passes them through here, and only the result is ever handed to the tracker. Three separate
// faults are modelled - a detection can be dropped entirely, its box can be reported in the wrong
// place, and the whole detection can arrive late. Turning any of these up is how the viewer and
// the headless runs answer "how well does this tracker hold up on a worse sensor," without ever
// touching the tracker itself.

Perturbation::Perturbation(PerturbationSettings settings, std::uint64_t seed)
    : settings_(settings), generator_(seed) {}

// Swaps in a new set of dials without disturbing the random generator underneath. The viewer
// calls this every frame so that dragging a slider changes the next frame's behaviour, while still
// drawing from the same continuing stream of randomness the run started with.
void Perturbation::set(PerturbationSettings settings) {
  settings_ = settings;
}

// Runs every detection in a frame through the three faults above, in a fixed order so that two
// runs with the same seed make the same decisions for the same detection every time. A detection
// is either dropped or kept and possibly moved; nothing here decides when it arrives, since that
// is available_time's job below.
std::vector<Detection> Perturbation::apply(const std::vector<Detection>& detections) {
  std::uniform_real_distribution<double> dropout_draw(0.0, 1.0);
  std::vector<Detection> kept;
  kept.reserve(detections.size());
  for (const Detection& detection : detections) {
    if (dropout_draw(generator_) < settings_.dropout_probability) continue;
    Detection perturbed = detection;
    if (settings_.position_noise_metres > 0.0) {
      std::normal_distribution<double> position_noise(0.0, settings_.position_noise_metres);
      perturbed.box.center_x += position_noise(generator_);
      perturbed.box.center_y += position_noise(generator_);
    }
    if (settings_.yaw_noise_radians > 0.0) {
      std::normal_distribution<double> yaw_noise(0.0, settings_.yaw_noise_radians);
      perturbed.box.yaw = wrap_angle(perturbed.box.yaw + yaw_noise(generator_));
    }
    kept.push_back(perturbed);
  }
  return kept;
}

// When a detection that was captured at this time would actually reach the tracker, once the
// sensor's own processing and transmission delay is accounted for. The tracker never sees a
// detection before this time, which is what lets a frame's own late detections turn up attached
// to a later frame instead.
std::int64_t Perturbation::available_time(std::int64_t capture_time_micros) const {
  return capture_time_micros + settings_.latency_micros;
}
