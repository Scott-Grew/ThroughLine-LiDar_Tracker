#include "replay.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>

// Turns a staged segment's labelled boxes into perturbed
// detections and steps a tracker over them, frame by frame.

namespace {

// One frame's detections, waiting to be handed to the tracker;
// carries its own capture time and pose for a late arrival.
struct PendingMeasurement {
  std::int64_t available_time_micros;
  std::int64_t capture_time_micros;
  std::vector<Detection> detections;
  Eigen::Matrix4d vehicle_to_world;
};

}  // namespace

// The value at the given fraction through the sorted samples,
// e.g. 0.5 and 0.99 for the p50 and p99 the summary line reports.
double TimingStats::percentile(double fraction) const {
  if (step_milliseconds.empty()) return 0.0;
  std::vector<double> sorted_milliseconds = step_milliseconds;
  std::sort(sorted_milliseconds.begin(), sorted_milliseconds.end());
  const std::size_t index = static_cast<std::size_t>(std::floor(
      fraction *
      static_cast<double>(sorted_milliseconds.size() - 1)));
  return sorted_milliseconds[index];
}

// Builds the tracker and perturbation for one run over segment,
// which must outlive this Replay.
Replay::Replay(const SegmentLog& segment, ReplaySettings settings)
    : segment_(segment),
      tracker_(settings.tracker),
      perturbation_(settings.perturbation, settings.seed) {
  confirmed_tracks_per_frame_.reserve(segment.frames.size());
}

// Perturbs each frame's boxes, queues them until they would
// arrive, and steps the tracker on whatever has arrived by then.
void Replay::run() {
  std::deque<PendingMeasurement> pending;

  // Falls back to a 10 Hz period when there is only one frame to
  // measure a gap from.
  const std::int64_t frame_period_micros =
      segment_.frames.size() >= 2
          ? segment_.frames[1].capture_time_micros -
                segment_.frames[0].capture_time_micros
          : 100000;
  const double frame_period_milliseconds =
      static_cast<double>(frame_period_micros) / 1000.0;

  for (const Frame& frame : segment_.frames) {
    std::vector<Detection> source_detections;
    source_detections.reserve(frame.ground_truth.size());
    for (const GroundTruthBox& truth : frame.ground_truth) {
      // Boxes with no lidar returns are excluded so the tracker
      // never sees knowledge no real detector could have.
      if (truth.lidar_points_in_box <= 0) continue;
      source_detections.push_back(
          Detection{truth.object_class, truth.box, 1.0f});
    }
    pending.push_back(PendingMeasurement{
        perturbation_.available_time(frame.capture_time_micros),
        frame.capture_time_micros,
        perturbation_.apply(source_detections),
        frame.vehicle_to_world});

    const auto step_start_time = std::chrono::steady_clock::now();
    while (!pending.empty() &&
           pending.front().available_time_micros <=
               frame.capture_time_micros) {
      tracker_.step(pending.front().capture_time_micros,
                    pending.front().detections,
                    pending.front().vehicle_to_world);
      pending.pop_front();
    }
    std::vector<Track> tracks_now =
        tracker_.confirmed_tracks_at(frame.capture_time_micros);
    const auto step_end_time = std::chrono::steady_clock::now();

    confirmed_tracks_per_frame_.push_back(std::move(tracks_now));

    const double step_milliseconds =
        std::chrono::duration<double, std::milli>(step_end_time -
                                                  step_start_time)
            .count();
    timing_.step_milliseconds.push_back(step_milliseconds);
    if (step_milliseconds > frame_period_milliseconds)
      timing_.overruns += 1;
  }
}

// The step timings recorded by the most recent run() call.
const TimingStats& Replay::timing() const {
  return timing_;
}

// The confirmed tracks from the most recent run() call, one
// entry per frame in segment order.
const std::vector<std::vector<Track>>&
Replay::confirmed_tracks_per_frame() const {
  return confirmed_tracks_per_frame_;
}
