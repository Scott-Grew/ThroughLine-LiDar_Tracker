#include "replay.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>

// This file walks a staged segment one frame at a time: it turns the
// frame's labelled boxes into detections, runs them through the
// perturbation stage, queues them with the time they would actually
// arrive, steps the tracker with everything that has arrived by this
// frame's time, and records the confirmed tracks for that frame.
// Nothing the tracker sees depends on the wall clock, so the same
// segment and seed always produce the same tracks; the clock is read
// only to time each step.
//
// Detections are always the ground-truth labels: boxes the sensor
// returned no points for are left out. Waymo labels objects it knows
// are there even when nothing came back from them, and its own
// scoring ignores those, so handing them to the tracker would give it
// knowledge no detector could have and then count every one of them
// against it as an object it invented.

namespace {

// One frame's detections, waiting between being perturbed and being
// handed to the tracker. It carries its own capture time and its own
// vehicle pose because by the time it arrives the replay may be on a
// later frame - a late detection must still be placed using the pose
// of the frame it was actually captured in.
struct PendingMeasurement {
  std::int64_t available_time_micros;
  std::int64_t capture_time_micros;
  std::vector<Detection> detections;
  Eigen::Matrix4d vehicle_to_world;
};

}  // namespace

// The value at a given fraction through the sorted samples, which is
// how the step timings become the p50 and p99 the summary line
// reports.
double TimingStats::percentile(double fraction) const {
  if (step_milliseconds.empty()) return 0.0;
  std::vector<double> sorted_milliseconds = step_milliseconds;
  std::sort(sorted_milliseconds.begin(), sorted_milliseconds.end());
  const std::size_t index = static_cast<std::size_t>(std::floor(
      fraction *
      static_cast<double>(sorted_milliseconds.size() - 1)));
  return sorted_milliseconds[index];
}

Replay::Replay(const SegmentLog& segment, ReplaySettings settings)
    : segment_(segment),
      tracker_(settings.tracker),
      perturbation_(settings.perturbation, settings.seed) {
  confirmed_tracks_per_frame_.reserve(segment.frames.size());
}

// For each frame: drop zero-point boxes, perturb, queue with arrival
// time, step the tracker on everything that has arrived, record what
// it confirmed, and time the step against the frame period.
void Replay::run() {
  std::deque<PendingMeasurement> pending;

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

const TimingStats& Replay::timing() const {
  return timing_;
}

const std::vector<std::vector<Track>>&
Replay::confirmed_tracks_per_frame() const {
  return confirmed_tracks_per_frame_;
}
