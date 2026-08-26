#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "export.hpp"
#include "log.hpp"
#include "predict.hpp"
#include "replay.hpp"
#include "viewer.hpp"

// This file is the front door: it turns command-line flags into the settings every other file
// runs on, then either runs a segment straight through headless or hands the same replay to the
// viewer on its own thread while it draws. Nothing decided here changes how the tracker, the
// perturbation stage or the metrics behave - it only chooses which of their existing knobs to
// turn, then prints or writes whatever they produced.

namespace {

// What to type to get any of the tracker's behaviour. Printed on --help and on a bad argument, so
// it is the one place every flag has to be kept in sync with the parsing below.
void print_usage() {
  std::cout << "Usage: tracker --segment PATH [--assign hungarian|greedy]\n"
                "               [--dropout PROBABILITY] [--noise METRES] [--latency MILLISECONDS]\n"
                "               [--sigma-position METRES] [--sigma-yaw RADIANS]\n"
                "               [--seed N] [--rate R] [--headless] [--record PATH]\n"
                "               [--export PATH] [--help]\n";
}

// The one report a run produces for a human to read: how wrong the tracker was per object class,
// by Waymo's own CLEAR accounting, and how fast the tracker itself ran. This is the training
// monitor's number, never the one that gets reported - that one only ever comes from Waymo's
// evaluator running on an export from this same run.
void print_summary(const TrackingMetrics& metrics, const TimingStats& timing) {
  for (const auto& [object_class, class_metrics] : metrics.per_class()) {
    std::cout << "class " << static_cast<int>(object_class)
              << " mota " << class_metrics.mota()
              << " motp " << class_metrics.motp()
              << " id_switches " << class_metrics.id_switches
              << " misses " << class_metrics.misses
              << " false_positives " << class_metrics.false_positives << "\n";
  }
  std::cout << "step_p50_ms " << timing.percentile(0.5)
            << " step_p99_ms " << timing.percentile(0.99)
            << " overruns " << timing.overruns
            << " frames " << timing.step_milliseconds.size() << "\n";
}

}  // namespace

// Parses the command line into a ReplaySettings, then runs one segment through the replay loop
// either headless or alongside the viewer, prints the summary, and writes an export file if one
// was asked for. Non-headless runs the replay on its own thread so the viewer can keep drawing
// while it works, and forces controls.quit once the viewer returns so a closed window always
// stops the replay thread rather than leaving it running with nothing left to show for it.
// Carries out one run: parses the command line, replays the segment, and prints what happened.
// Kept separate from main so that every failure it can raise - a segment file that is missing or
// truncated, a recording that cannot be written - arrives somewhere that can turn it into a line
// of text and an exit code, instead of ending the process with an abort and throwing the
// explanation away.
int run(int argument_count, char** arguments) {
  std::string segment_path;
  bool segment_path_given = false;
  std::string export_path;
  bool export_requested = false;
  std::string record_path;
  bool record_requested = false;
  ReplaySettings settings;
  settings.tracker.noise.sigma_measurement_position = 0.1;
  settings.tracker.noise.sigma_measurement_yaw = 0.02;

  for (int argument_index = 1; argument_index < argument_count; ++argument_index) {
    std::string argument = arguments[argument_index];

    if (argument == "--help") {
      print_usage();
      return 0;
    }

    if (argument == "--segment" && argument_index + 1 < argument_count) {
      segment_path = arguments[++argument_index];
      segment_path_given = true;
      continue;
    }

    if (argument == "--assign" && argument_index + 1 < argument_count) {
      std::string value = arguments[++argument_index];
      if (value == "hungarian") {
        settings.tracker.assignment = AssignmentMethod::Hungarian;
        continue;
      }
      if (value == "greedy") {
        settings.tracker.assignment = AssignmentMethod::Greedy;
        continue;
      }
      print_usage();
      return 2;
    }

    if (argument == "--dropout" && argument_index + 1 < argument_count) {
      settings.perturbation.dropout_probability = std::stod(arguments[++argument_index]);
      continue;
    }

    if (argument == "--noise" && argument_index + 1 < argument_count) {
      settings.perturbation.position_noise_metres = std::stod(arguments[++argument_index]);
      continue;
    }

    if (argument == "--latency" && argument_index + 1 < argument_count) {
      double latency_milliseconds = std::stod(arguments[++argument_index]);
      settings.perturbation.latency_micros = static_cast<std::int64_t>(latency_milliseconds * 1000.0);
      continue;
    }

    if (argument == "--sigma-position" && argument_index + 1 < argument_count) {
      settings.tracker.noise.sigma_measurement_position = std::stod(arguments[++argument_index]);
      continue;
    }

    if (argument == "--sigma-yaw" && argument_index + 1 < argument_count) {
      settings.tracker.noise.sigma_measurement_yaw = std::stod(arguments[++argument_index]);
      continue;
    }

    if (argument == "--seed" && argument_index + 1 < argument_count) {
      settings.seed = std::stoull(arguments[++argument_index]);
      continue;
    }

    if (argument == "--rate" && argument_index + 1 < argument_count) {
      settings.rate = std::stod(arguments[++argument_index]);
      continue;
    }

    if (argument == "--headless") {
      settings.headless = true;
      continue;
    }

    if (argument == "--record" && argument_index + 1 < argument_count) {
      record_path = arguments[++argument_index];
      record_requested = true;
      continue;
    }

    if (argument == "--export" && argument_index + 1 < argument_count) {
      export_path = arguments[++argument_index];
      export_requested = true;
      continue;
    }

    print_usage();
    return 2;
  }

  if (!segment_path_given) {
    print_usage();
    return 2;
  }

  SegmentLog segment = read_segment_log(segment_path);
  ConstantTurnRatePredictor predictor(settings.tracker.noise);
  Replay replay(segment, settings, predictor);
  SnapshotExchange exchange;
  LiveControls controls;

  if (settings.headless) {
    replay.run(exchange, controls);
    print_summary(replay.metrics(), replay.timing());
  } else {
    std::thread replay_thread([&replay, &exchange, &controls]() { replay.run(exchange, controls); });
    run_viewer(exchange, controls);
    controls.quit = true;
    replay_thread.join();
    print_summary(replay.metrics(), replay.timing());
  }

  if (export_requested) {
    export_tracks(export_path, segment, replay.confirmed_tracks_per_frame());
  }

  if (record_requested) {
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
