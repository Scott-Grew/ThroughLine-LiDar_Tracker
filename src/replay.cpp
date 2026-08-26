#include "replay.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <thread>

// This file is the clock the rest of the project runs on. It walks a staged segment one frame at
// a time, decides which detections that frame contributes, runs them through the perturbation
// stage, feeds whatever has actually "arrived" by this frame's time into the tracker, scores the
// result against ground truth, and hands a snapshot of all of that to whichever thread is drawing
// the viewer. It is also what makes a run reproducible: the same segment and the same seed walk
// through exactly the same detections, drops, noise and arrival times whether the run is paced to
// real time or driven through as fast as possible, because nothing that changes what the tracker
// sees is allowed to depend on the wall clock.
//
// When the run is set to take its detections from the labels rather than from a detector, boxes
// that the sensor returned no points for are left out. Waymo labels objects it knows are there
// even when nothing came back from them, and its own scoring ignores those, so handing them to the
// tracker would give it knowledge no detector could have and then count every one of them against
// it as an object it invented.

namespace {

// One frame's detections, sitting in the queue between being perturbed and being handed to the
// tracker. It carries its own capture time and its own vehicle pose because by the time its
// arrival time comes due, the replay loop may already be looking at a later frame's pose - a late
// detection must still be placed using the pose of the frame it was actually captured in.
struct PendingMeasurement {
  std::int64_t available_time_micros;
  std::int64_t capture_time_micros;
  std::vector<Detection> detections;
  Eigen::Matrix4d vehicle_to_world;
};

}

// The value at a given fraction through the sorted samples seen so far, which is how the step
// timings below turn into the p50 and p99 the viewer and the summary line report. An empty run
// has no timings to report a percentile of.
double TimingStats::percentile(double fraction) const {
  if (step_milliseconds.empty()) return 0.0;
  std::vector<double> sorted_milliseconds = step_milliseconds;
  std::sort(sorted_milliseconds.begin(), sorted_milliseconds.end());
  const std::size_t index = static_cast<std::size_t>(std::floor(fraction * static_cast<double>(sorted_milliseconds.size() - 1)));
  return sorted_milliseconds[index];
}

// The buffer the replay thread is currently free to fill in. It is never the buffer the viewer is
// reading from or the one most recently handed to it, which is what makes this lock-free: the
// writer and the reader can never collide on the same slot.
Snapshot& SnapshotExchange::writable() {
  return buffers_[writing_];
}

// Hands the buffer just filled in over to the reader and takes back whichever buffer the reader is
// no longer looking at, marking the handed-over one as fresh. This is the only place the writer
// and the reader touch the same atomic, and it is a single exchange rather than a lock, which is
// what keeps a slow or paused viewer from ever blocking the replay thread.
void SnapshotExchange::publish() {
  writing_ = published_.exchange(writing_ | kFreshFlag, std::memory_order_acq_rel) & ~kFreshFlag;
}

// Hands the viewer the newest snapshot available. If nothing new has been published since the
// last call, the same buffer already being read is handed back unchanged, so a viewer running
// faster than the replay thread simply redraws the same frame rather than tearing into a buffer
// that is still being written.
const Snapshot* SnapshotExchange::acquire() {
  if (!(published_.load(std::memory_order_acquire) & kFreshFlag)) return &buffers_[reading_];
  reading_ = published_.exchange(reading_, std::memory_order_acq_rel) & ~kFreshFlag;
  return &buffers_[reading_];
}

Replay::Replay(const SegmentLog& segment, ReplaySettings settings, const Predictor& predictor)
    : segment_(segment),
      settings_(settings),
      predictor_(predictor),
      tracker_(settings.tracker),
      perturbation_(settings.perturbation, settings.seed) {
  confirmed_tracks_per_frame_.reserve(segment.frames.size());
}

// Walks the whole segment once, frame by frame. Each frame's ground truth or detector output is
// perturbed and queued with the time it would actually become available, whatever in the queue has
// become available by this frame's capture time is stepped through the tracker, and the confirmed
// result is scored, recorded and published for the viewer. Wall-clock time is only ever read to
// decide how long to sleep and how a step's duration compares to the pacing budget - never to
// decide what the tracker sees - so a headless run and a paced run at any rate walk through the
// exact same sequence of tracker calls and produce the exact same confirmed tracks.
void Replay::run(SnapshotExchange& exchange, LiveControls& controls) {
  std::deque<PendingMeasurement> pending;

  const std::int64_t frame_period_micros = segment_.frames.size() >= 2
                                                ? segment_.frames[1].capture_time_micros - segment_.frames[0].capture_time_micros
                                                : 100000;
  const double pacing_period_seconds = static_cast<double>(frame_period_micros) / 1e6 / settings_.rate;
  const auto start_time = std::chrono::steady_clock::now();

  for (std::size_t frame_index = 0; frame_index < segment_.frames.size(); ++frame_index) {
    const Frame& frame = segment_.frames[frame_index];

    while (controls.paused.load() && !controls.quit.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (controls.quit.load()) break;

    PerturbationSettings live_perturbation;
    live_perturbation.dropout_probability = controls.dropout_probability.load();
    live_perturbation.position_noise_metres = controls.position_noise_metres.load();
    live_perturbation.latency_micros = controls.latency_micros.load();
    live_perturbation.yaw_noise_radians = settings_.perturbation.yaw_noise_radians;
    perturbation_.set(live_perturbation);

    std::vector<Detection> source_detections;
    if (settings_.source == DetectionSource::GroundTruth) {
      source_detections.reserve(frame.ground_truth.size());
      for (const GroundTruthBox& truth : frame.ground_truth) {
        if (truth.lidar_points_in_box <= 0) continue;
        source_detections.push_back(Detection{truth.object_class, truth.box, 1.0f});
      }
    } else {
      source_detections = frame.detections;
    }

    pending.push_back(PendingMeasurement{perturbation_.available_time(frame.capture_time_micros),
                                          frame.capture_time_micros,
                                          perturbation_.apply(source_detections),
                                          frame.vehicle_to_world});

    const auto step_start_time = std::chrono::steady_clock::now();
    while (!pending.empty() && pending.front().available_time_micros <= frame.capture_time_micros) {
      tracker_.step(pending.front().capture_time_micros, pending.front().detections, pending.front().vehicle_to_world);
      pending.pop_front();
    }
    std::vector<Track> tracks_now = tracker_.confirmed_tracks_at(frame.capture_time_micros);
    const auto step_end_time = std::chrono::steady_clock::now();

    metrics_.update(tracks_now, frame.ground_truth, frame.vehicle_to_world);
    confirmed_tracks_per_frame_.push_back(tracks_now);

    const double step_milliseconds = std::chrono::duration<double, std::milli>(step_end_time - step_start_time).count();
    timing_.step_milliseconds.push_back(step_milliseconds);
    if (step_milliseconds > pacing_period_seconds * 1000.0) timing_.overruns += 1;

    Snapshot& snapshot = exchange.writable();
    snapshot.frame_index = frame_index;
    snapshot.frame = &frame;
    snapshot.tracks = tracks_now;
    snapshot.predictions.clear();
    snapshot.predictions.reserve(tracks_now.size());
    for (const Track& track : tracks_now)
      snapshot.predictions.push_back(predictor_.predict(track, settings_.prediction_horizon_seconds, settings_.prediction_step_seconds));
    snapshot.metrics = metrics_.per_class();
    snapshot.step_p50_milliseconds = timing_.percentile(0.5);
    snapshot.step_p99_milliseconds = timing_.percentile(0.99);
    snapshot.overruns = timing_.overruns;
    exchange.publish();

    if (!settings_.headless) {
      const std::chrono::duration<double> elapsed_budget(pacing_period_seconds * static_cast<double>(frame_index + 1));
      std::this_thread::sleep_until(start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(elapsed_budget));
    }
  }
}

const TrackingMetrics& Replay::metrics() const {
  return metrics_;
}

const TimingStats& Replay::timing() const {
  return timing_;
}

const std::vector<std::vector<Track>>& Replay::confirmed_tracks_per_frame() const {
  return confirmed_tracks_per_frame_;
}
