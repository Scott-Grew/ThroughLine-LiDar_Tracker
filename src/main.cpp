#include <iostream>
#include <string>
#include <thread>

#include "export.hpp"
#include "log.hpp"
#include "predict.hpp"
#include "replay.hpp"
#include "viewer.hpp"

namespace {

void print_usage() {
  std::cout << "Usage: tracker --segment PATH [--source gt|det] [--assign hungarian|greedy]\n"
                "               [--dropout PROBABILITY] [--noise METRES] [--latency MILLISECONDS]\n"
                "               [--seed N] [--rate R] [--headless] [--export PATH] [--help]\n";
}

void print_summary(const TrackingMetrics& metrics, const TimingStats& timing) {
  for (const auto& [object_class, class_metrics] : metrics.per_class()) {
    std::cout << "class " << static_cast<int>(object_class)
              << " mota " << class_metrics.mota()
              << " motp " << class_metrics.motp() << "\n";
  }
  std::cout << "p50 " << timing.percentile(0.5) << "ms"
            << " p99 " << timing.percentile(0.99) << "ms"
            << " overruns " << timing.overruns << "\n";
}

}  // namespace

int main(int argument_count, char** arguments) {
  std::string segment_path;
  bool segment_path_given = false;
  std::string export_path;
  bool export_requested = false;
  ReplaySettings settings;

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

    if (argument == "--source" && argument_index + 1 < argument_count) {
      std::string value = arguments[++argument_index];
      if (value == "gt") {
        settings.source = DetectionSource::GroundTruth;
        continue;
      }
      if (value == "det") {
        settings.source = DetectionSource::Detector;
        continue;
      }
      print_usage();
      return 2;
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
    replay_thread.join();
    print_summary(replay.metrics(), replay.timing());
  }

  if (export_requested) {
    export_tracks(export_path, segment, replay.confirmed_tracks_per_frame());
  }

  return 0;
}
