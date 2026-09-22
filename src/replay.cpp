// Turns a staged segment's labelled boxes into perturbed
// detections and steps a tracker over them, frame by frame.

#include "replay.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>

namespace {

// One frame's detections, waiting to be handed to the tracker;
// carries its own capture time and pose for a late arrival.
struct PendingMeasurement {
  std::int64_t available_time_micros;
  std::int64_t capture_time_micros;
  std::vector<Detection> detections;
  Eigen::Matrix4d vehicle_to_world;
};

// Frame period assumed for a one-frame segment, Waymo's 10 Hz.
constexpr std::int64_t kFallbackFramePeriodMicros = 100'000;

}  // namespace

double TimingStats::percentile(double fraction) const {
  if (step_milliseconds.empty()) return 0.0;
  std::vector<double> sorted_milliseconds = step_milliseconds;
  std::sort(sorted_milliseconds.begin(), sorted_milliseconds.end());
  const double last_index = static_cast<double>(sorted_milliseconds.size() - 1);
  // The percentile is the sample at index floor(fraction * (n - 1)).
  const std::size_t index =
      static_cast<std::size_t>(std::floor(fraction * last_index));
  return sorted_milliseconds[index];
}

Replay::Replay(const SegmentLog& segment, ReplaySettings settings)
    : segment_(segment),
      tracker_(settings.tracker),
      perturbation_(settings.perturbation, settings.seed) {
  confirmed_tracks_per_frame_.reserve(segment.frames.size());
}

void Replay::run() {
  std::deque<PendingMeasurement> pending_measurements;

  const std::int64_t frame_period_micros =
      segment_.frames.size() >= 2 ? segment_.frames[1].capture_time_micros -
                                        segment_.frames[0].capture_time_micros
                                  : kFallbackFramePeriodMicros;
  const double frame_period_milliseconds =
      static_cast<double>(frame_period_micros) / 1000.0;

  for (const Frame& frame : segment_.frames) {
    std::vector<Detection> ground_truth_detections;
    ground_truth_detections.reserve(frame.ground_truth.size());
    for (const GroundTruthBox& ground_truth_box : frame.ground_truth) {
      if (ground_truth_box.lidar_points_in_box <= 0) continue;
      ground_truth_detections.push_back(
          Detection{ground_truth_box.object_class, ground_truth_box.box, 1.0f});
    }
    pending_measurements.push_back(PendingMeasurement{
        perturbation_.available_time(frame.capture_time_micros),
        frame.capture_time_micros, perturbation_.apply(ground_truth_detections),
        frame.vehicle_to_world});

    const auto step_start_time = std::chrono::steady_clock::now();
    while (!pending_measurements.empty() &&
           pending_measurements.front().available_time_micros <=
               frame.capture_time_micros) {
      tracker_.step(pending_measurements.front().capture_time_micros,
                    pending_measurements.front().detections,
                    pending_measurements.front().vehicle_to_world);
      pending_measurements.pop_front();
    }
    std::vector<Track> confirmed_tracks =
        tracker_.confirmed_tracks_at(frame.capture_time_micros);
    const auto step_end_time = std::chrono::steady_clock::now();

    confirmed_tracks_per_frame_.push_back(std::move(confirmed_tracks));

    const std::chrono::duration<double, std::milli> step_duration =
        step_end_time - step_start_time;
    const double step_milliseconds = step_duration.count();
    timing_.step_milliseconds.push_back(step_milliseconds);
    if (step_milliseconds > frame_period_milliseconds) timing_.overruns += 1;
  }
}

const TimingStats& Replay::timing() const {
  return timing_;
}

const std::vector<std::vector<Track>>& Replay::confirmed_tracks_per_frame()
    const {
  return confirmed_tracks_per_frame_;
}
