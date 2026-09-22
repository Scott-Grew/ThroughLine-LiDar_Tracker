// Command-line entry point: turns flags into settings and replays a
// segment. Waymo's evaluator scores the export; nothing here does.

#include <iostream>
#include <stdexcept>
#include <string>

#include <CLI/CLI.hpp>

#include "export.hpp"
#include "log.hpp"
#include "predict.hpp"
#include "recording.hpp"
#include "replay.hpp"

namespace {

// How far ahead the recording draws each track's predicted path, and
// the spacing of its points. Chosen.
constexpr double kPredictionHorizonSeconds = 3.0;
constexpr double kPredictionStepSeconds = 0.1;

// Prints step p50 and p99 in milliseconds, the overrun count and
// the frame count to stdout.
void print_summary(const TimingStats& timing) {
  std::cout << "step_p50_ms " << timing.percentile(0.5) << " step_p99_ms "
            << timing.percentile(0.99) << " overruns " << timing.overruns
            << " frames " << timing.step_milliseconds.size() << "\n";
}

}  // namespace

// Parses flags, replays the segment, then writes the export and
// recording if asked. Throws on failure; main reports it.
int run(int argument_count, char** arguments) {
  CLI::App app{"Waymo multi-object tracker"};
  std::string segment_path;
  std::string export_path;
  std::string record_path;
  std::string assignment_name = "hungarian";
  double latency_milliseconds = 0.0;
  ReplaySettings settings;

  app.add_option("--segment", segment_path, "staged .trklog to replay")
      ->required();
  app.add_option("--assign", assignment_name, "hungarian or greedy")
      ->check(CLI::IsMember({"hungarian", "greedy"}));
  app.add_option("--dropout", settings.perturbation.dropout_probability,
                 "probability a detection is dropped");
  app.add_option("--noise", settings.perturbation.position_noise_metres,
                 "position noise sigma, metres");
  app.add_option("--latency", latency_milliseconds,
                 "detection latency, milliseconds");
  app.add_option("--sigma-position",
                 settings.tracker.noise.sigma_measurement_position,
                 "measurement position sigma, metres (default: the "
                 "segment's measured box jitter)");
  app.add_option("--sigma-yaw", settings.tracker.noise.sigma_measurement_yaw,
                 "measurement yaw sigma, radians (default: the "
                 "segment's measured box jitter)");
  app.add_option("--seed", settings.seed, "perturbation random seed");
  app.add_option("--record", record_path, "write an .mcap recording");
  app.add_option("--export", export_path, "write confirmed tracks as CSV");

  CLI11_PARSE(app, argument_count, arguments);
  settings.tracker.assignment = assignment_name == "greedy"
                                    ? AssignmentMethod::Greedy
                                    : AssignmentMethod::Hungarian;
  settings.perturbation.latency_micros =
      static_cast<std::int64_t>(latency_milliseconds * 1000.0);

  const SegmentLog segment = read_segment_log(segment_path);
  if (app.count("--sigma-position") == 0)
    settings.tracker.noise.sigma_measurement_position =
        segment.measured_position_sigma;
  if (app.count("--sigma-yaw") == 0)
    settings.tracker.noise.sigma_measurement_yaw = segment.measured_yaw_sigma;
  Replay replay(segment, settings);
  replay.run();
  print_summary(replay.timing());

  if (!export_path.empty())
    export_tracks(export_path, segment, replay.confirmed_tracks_per_frame());
  if (!record_path.empty()) {
    const ConstantTurnRatePredictor predictor(settings.tracker.noise);
    save_replay_recording(record_path, segment,
                          replay.confirmed_tracks_per_frame(), predictor,
                          kPredictionHorizonSeconds, kPredictionStepSeconds);
  }
  return 0;
}

// Turns anything the run raises into a message on the error stream
// and a non-zero exit code.
int main(int argument_count, char** arguments) {
  try {
    return run(argument_count, arguments);
  } catch (const std::exception& failure) {
    std::cerr << "tracker: " << failure.what() << "\n";
    return 1;
  }
}

// clang-format off
// Data flow of one run.
//
//   stage/stage_segment.py          Waymo parquet -> .trklog
//            |
//            v
//   read_segment_log                SegmentLog: frames of pose, points, boxes
//            |
//            v
//   Replay::run, for each frame:
//     ground truth boxes            vehicle frame, at least one lidar point
//       -> Perturbation::apply      dropout, position noise
//       -> queue until arrival      latency
//       -> Tracker::step            boxes to world frame, predict, assign per
//                                   class, update, create, delete
//       -> confirmed_tracks_at      confirmed tracks at the frame's time
//            |
//            +--> export_tracks           CSV, back in each frame's vehicle frame
//            |      -> eval/run_official.sh     Waymo's evaluator
//            +--> save_replay_recording   .mcap topics /tf, /points, /scene
// clang-format on
