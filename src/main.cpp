// Runs the pipeline live, one frame at a time: read a staged frame at its
// capture pace, track it, publish its confirmed tracks, take the next.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <exception>
#include <ratio>
#include <thread>
#include <utility>
#include <vector>
#include <CLI/CLI.hpp>
#include <Eigen/Dense>
#include "filter.hpp"
#include "log.hpp"
#include "scene.hpp"
#include "tracker.hpp"
#include "types.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Micros = std::chrono::microseconds;

// Writes one row per confirmed track per frame, boxes in the frame's own
// vehicle frame, which is what Waymo's evaluator compares against.
class CsvWriter {
 public:
  explicit CsvWriter(const std::string& path) : stream_(path) {
    if (!stream_) throw std::runtime_error("cannot open " + path);
    stream_ << "frame_timestamp_micros,track_id,object_class,center_x,"
               "center_y,center_z,length,width,height,yaw\n";
  }

  void write(const Frame& frame, const std::vector<Track>& tracks) {
    const Eigen::Matrix4d vehicle_from_world = frame.vehicle_to_world.inverse();
    const double ego_yaw = vehicle_yaw(frame.vehicle_to_world);
    for (const Track& track : tracks) {
      const Eigen::Vector2d position = track.state.position();
      const Eigen::Vector4d world(position.x(), position.y(),
                                  track.state.center_z, 1.0);
      const Eigen::Vector4d local = vehicle_from_world * world;
      const double yaw = wrap_angle(track.state.yaw() - ego_yaw);
      stream_ << frame.capture_time_micros << ',' << track.track_id << ','
              << static_cast<int>(track.object_class) << ',' << local.x() << ','
              << local.y() << ',' << local.z() << ',' << track.state.length
              << ',' << track.state.width << ',' << track.state.height << ','
              << yaw << '\n';
    }
  }

 private:
  std::ofstream stream_;
};

// Wall-clock time of each frame's tracking step, in milliseconds.
struct StepTimes {
  std::vector<double> milliseconds;

  // The sample at index floor(fraction * (n - 1)) of the sorted times.
  double percentile(double fraction) const {
    if (milliseconds.empty()) return 0.0;
    std::vector<double> sorted = milliseconds;
    std::ranges::sort(sorted);
    const auto last = static_cast<double>(sorted.size() - 1);
    return sorted[static_cast<std::size_t>(std::floor(fraction * last))];
  }

  std::size_t overruns(double frame_period_milliseconds) const {
    const auto slow = [=](double step) {
      return step > frame_period_milliseconds;
    };
    return static_cast<std::size_t>(std::ranges::count_if(milliseconds, slow));
  }
};

// Holds the loop to the frames' own pace: frame k is released when as much
// wall time has passed since the first frame as capture time has.
class Pacer {
 public:
  void wait_for(std::int64_t capture_time_micros) {
    if (!start_) start_.emplace(Clock::now(), capture_time_micros);
    const Micros elapsed(capture_time_micros - start_->second);
    std::this_thread::sleep_until(start_->first + elapsed);
  }

 private:
  std::optional<std::pair<Clock::time_point, std::int64_t>> start_;
};

int run(int argument_count, char* const* arguments) {
  CLI::App app{"Live multi-object tracker on a staged Waymo segment"};
  std::string segment_path;
  std::string csv_path;
  SceneOptions scene_options;
  bool paced = false;
  TrackerSettings settings;
  app.add_option("--segment", segment_path, "staged .mcap segment to track")
      ->required();
  app.add_option("--live", scene_options.live_port,
                 "serve the scene to Lichtblick on this port, at the "
                 "frames' own pace");
  app.add_option("--record", scene_options.recording_path,
                 "also write the scene to this .mcap file");
  app.add_option("--export", csv_path, "write confirmed tracks as CSV");
  app.add_option("--sigma-position", settings.noise.sigma_measurement_position,
                 "measurement position sigma, metres (default: the "
                 "detector's measured error on the segment)");
  app.add_option("--sigma-yaw", settings.noise.sigma_measurement_yaw,
                 "measurement yaw sigma, radians (default: the detector's "
                 "measured error on the segment)");
  CLI11_PARSE(app, argument_count, arguments);
  paced = scene_options.live_port != 0;

  SegmentReader reader(segment_path);
  if (app.count("--sigma-position") == 0)
    settings.noise.sigma_measurement_position = reader.info().position_sigma;
  if (app.count("--sigma-yaw") == 0)
    settings.noise.sigma_measurement_yaw = reader.info().yaw_sigma;

  Tracker tracker(settings);
  std::optional<CsvWriter> csv;
  if (!csv_path.empty()) csv.emplace(csv_path);
  std::optional<Scene> scene;
  if (paced || !scene_options.recording_path.empty())
    scene.emplace(scene_options, settings.noise);
  if (paced)
    std::cout << "serving ws://127.0.0.1:" << scene_options.live_port
              << " for Lichtblick\n";

  Pacer pacer;
  StepTimes step_times;
  std::vector<std::int64_t> capture_times;
  while (const std::optional<Frame> frame = reader.next_frame()) {
    if (paced) pacer.wait_for(frame->capture_time_micros);
    const Clock::time_point start = Clock::now();
    tracker.step(*frame);
    const std::vector<Track> tracks =
        tracker.confirmed_at(frame->capture_time_micros);
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - start;
    step_times.milliseconds.push_back(elapsed.count());
    capture_times.push_back(frame->capture_time_micros);

    if (csv) csv->write(*frame, tracks);
    if (scene) scene->publish(*frame, tracks);
  }
  if (scene) scene->close();

  const double frame_period_milliseconds =
      capture_times.size() >= 2
          ? static_cast<double>(capture_times[1] - capture_times[0]) / 1000.0
          : 100.0;
  std::cout << "step_p50_ms " << step_times.percentile(0.5) << " step_p99_ms "
            << step_times.percentile(0.99) << " overruns "
            << step_times.overruns(frame_period_milliseconds) << " frames "
            << step_times.milliseconds.size() << "\n";
  return 0;
}

}  // namespace

int main(int argument_count, char** arguments) {
  try {
    return run(argument_count, arguments);
  } catch (const std::exception& failure) {
    std::cerr << "tracker: " << failure.what() << "\n";
    return 1;
  }
}
