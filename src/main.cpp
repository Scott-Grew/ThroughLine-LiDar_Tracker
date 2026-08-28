#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <CLI/CLI.hpp>

#include "export.hpp"
#include "log.hpp"
#include "predict.hpp"
#include "replay.hpp"
#include "viewer.hpp"

// This file is the front door: it turns command-line flags into the settings every other file
// runs on, then either runs a segment straight through headless or hands the same replay to the
// viewer on its own thread while it draws. Nothing decided here changes how the tracker or the
// perturbation stage behave - it only chooses which of their existing knobs to turn, then prints
// or writes whatever they produced. Scores come from Waymo's evaluator on the export, never from
// here.

namespace {

// The one report a run prints: how fast the tracker itself ran.
void print_summary(const TimingStats& timing) {
  std::cout << "step_p50_ms " << timing.percentile(0.5)
            << " step_p99_ms " << timing.percentile(0.99)
            << " overruns " << timing.overruns
            << " frames " << timing.step_milliseconds.size() << "\n";
}

}  // namespace

// Parses the command line into a ReplaySettings, then runs one segment through the replay loop
// either headless or alongside the viewer, prints the timing summary, and writes an export or a
// recording if asked. Kept separate from main so every failure it can raise arrives somewhere
// that turns it into a line of text and an exit code.
int run(int argument_count, char** arguments) {
  CLI::App app{"Waymo multi-object tracker"};
  std::string segment_path;
  std::string export_path;
  std::string record_path;
  std::string assignment_name = "hungarian";
  double latency_milliseconds = 0.0;
  ReplaySettings settings;
  settings.tracker.noise.sigma_measurement_position = 0.1;
  settings.tracker.noise.sigma_measurement_yaw = 0.02;

  app.add_option("--segment", segment_path, "staged .trklog to replay")->required();
  app.add_option("--assign", assignment_name, "hungarian or greedy")->check(CLI::IsMember({"hungarian", "greedy"}));
  app.add_option("--dropout", settings.perturbation.dropout_probability, "probability a detection is dropped");
  app.add_option("--noise", settings.perturbation.position_noise_metres, "position noise sigma, metres");
  app.add_option("--latency", latency_milliseconds, "detection latency, milliseconds");
  app.add_option("--sigma-position", settings.tracker.noise.sigma_measurement_position, "measurement position sigma, metres");
  app.add_option("--sigma-yaw", settings.tracker.noise.sigma_measurement_yaw, "measurement yaw sigma, radians");
  app.add_option("--seed", settings.seed, "perturbation random seed");
  app.add_option("--rate", settings.rate, "playback rate relative to real time");
  app.add_flag("--headless", settings.headless, "run without the viewer");
  app.add_option("--record", record_path, "write an .mcap recording");
  app.add_option("--export", export_path, "write confirmed tracks as CSV");

  CLI11_PARSE(app, argument_count, arguments);

  settings.tracker.assignment = assignment_name == "greedy" ? AssignmentMethod::Greedy : AssignmentMethod::Hungarian;
  settings.perturbation.latency_micros = static_cast<std::int64_t>(latency_milliseconds * 1000.0);

  SegmentLog segment = read_segment_log(segment_path);
  ConstantTurnRatePredictor predictor(settings.tracker.noise);
  Replay replay(segment, settings, predictor);
  SnapshotExchange exchange;
  LiveControls controls;

  if (settings.headless) {
    replay.run(exchange, controls);
  } else {
    std::thread replay_thread([&replay, &exchange, &controls]() { replay.run(exchange, controls); });
    run_viewer(exchange, controls);
    controls.quit = true;
    replay_thread.join();
  }
  print_summary(replay.timing());

  if (!export_path.empty()) export_tracks(export_path, segment, replay.confirmed_tracks_per_frame());
  if (!record_path.empty()) {
    save_replay_recording(record_path, segment, replay.confirmed_tracks_per_frame(), predictor,
                          settings.prediction_horizon_seconds, settings.prediction_step_seconds);
  }
  return 0;
}

// Turns anything the run raises into a message on the error stream and a non-zero exit code.
int main(int argument_count, char** arguments) {
  try {
    return run(argument_count, arguments);
  } catch (const std::exception& failure) {
    std::cerr << "tracker: " << failure.what() << "\n";
    return 1;
  }
}
