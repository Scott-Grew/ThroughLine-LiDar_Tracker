#include "viewer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <rerun.hpp>

// This file is the only place the project draws anything. Rerun owns every 3d shape: the lidar
// points, the ego box, each track's box, its history trail, its predicted path and its
// uncertainty ellipse, plus the running metrics and timing plots. A small ImGui window sits next
// to Rerun's own window and does only two things - it lets the perturbation sliders move the
// atomics in LiveControls, and it prints the same numbers the headless summary prints so the
// person watching does not have to read a terminal. Every position handed to Rerun is first
// shifted by recording_origin's translation, because a raw Waymo world coordinate can be tens of
// thousands of metres from zero and would burn through a float's precision long before it reached
// the screen; log_frame is the one function both run_viewer and save_replay_recording call, so a
// live session and a saved .rrd are always built from exactly the same drawing code.

namespace {

constexpr double kPointRadius = 0.03;
constexpr int kEllipseSegments = 32;
constexpr double kEllipseSigma = 2.0;
constexpr float kHeightRampLow = -2.0f;
constexpr float kHeightRampHigh = 4.0f;

// The one colour a class is ever drawn in, everywhere a track of that class appears: its box, its
// history trail, its predicted path and its uncertainty ellipse all share this so an object reads
// as the same object across every one of those entities.
rerun::Color class_color(ObjectClass object_class) {
  switch (object_class) {
    case ObjectClass::Vehicle: return rerun::Color(0, 150, 255);
    case ObjectClass::Pedestrian: return rerun::Color(255, 140, 0);
    case ObjectClass::Cyclist: return rerun::Color(0, 220, 120);
  }
  return rerun::Color(255, 255, 255);
}

// The name a class is filed under in the metrics tree and printed in the controls window.
// Nothing about tracking uses this string; it only exists so a human reading the viewer can tell
// the three classes apart.
std::string class_name(ObjectClass object_class) {
  switch (object_class) {
    case ObjectClass::Vehicle: return "vehicle";
    case ObjectClass::Pedestrian: return "pedestrian";
    case ObjectClass::Cyclist: return "cyclist";
  }
  return "unknown";
}

// The translation of frame 0's vehicle pose, in Waymo's world frame. Every position this file
// ever hands to Rerun has this subtracted from it first, because a segment's world coordinates can
// sit tens of thousands of metres from the origin and a float only has so many bits of precision
// to spend - without this shift, a lidar point and a track box a few metres apart could round to
// the same float and jitter on screen.
Eigen::Vector3d recording_origin(const SegmentLog& segment) {
  return segment.frames.front().vehicle_to_world.block<3, 1>(0, 3);
}

// Turns a height above the ground into a colour on a blue-to-green-to-yellow ramp, clamped to
// [kHeightRampLow, kHeightRampHigh] first so a stray high or low point cannot pull the ramp's
// range around for every other point in the same cloud.
rerun::Color height_ramp_color(float height) {
  const float clamped_height = std::clamp(height, kHeightRampLow, kHeightRampHigh);
  const float ramp_fraction = (clamped_height - kHeightRampLow) / (kHeightRampHigh - kHeightRampLow);

  const Eigen::Vector3f blue(0.0f, 0.0f, 1.0f);
  const Eigen::Vector3f green(0.0f, 1.0f, 0.0f);
  const Eigen::Vector3f yellow(1.0f, 1.0f, 0.0f);

  Eigen::Vector3f ramp_color;
  if (ramp_fraction < 0.5f) {
    const float local_fraction = ramp_fraction / 0.5f;
    ramp_color = blue + local_fraction * (green - blue);
  } else {
    const float local_fraction = (ramp_fraction - 0.5f) / 0.5f;
    ramp_color = green + local_fraction * (yellow - green);
  }
  return rerun::Color(static_cast<std::uint8_t>(ramp_color.x() * 255.0f),
                       static_cast<std::uint8_t>(ramp_color.y() * 255.0f),
                       static_cast<std::uint8_t>(ramp_color.z() * 255.0f));
}

// Builds the closed ring of points that draws as an uncertainty ellipse: kEllipseSigma standard
// deviations along each eigenvector of the given 2x2 covariance, centred on the given position,
// at the given height. This is the shape the viewer draws around a track's final predicted
// position, so a wide ellipse there means the filter has stopped trusting its own guess.
std::vector<rerun::Vec3D> ellipse_points(const Eigen::Vector2d& center, const Eigen::Matrix2d& covariance,
                                          double height, const Eigen::Vector3d& origin) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(covariance);
  const Eigen::Vector2d eigenvalues = eigen_solver.eigenvalues();
  const Eigen::Matrix2d eigenvectors = eigen_solver.eigenvectors();

  std::vector<rerun::Vec3D> points;
  points.reserve(kEllipseSegments + 1);
  for (int segment_index = 0; segment_index <= kEllipseSegments; ++segment_index) {
    const double angle = 2.0 * M_PI * static_cast<double>(segment_index) / static_cast<double>(kEllipseSegments);
    const Eigen::Vector2d unit_circle(std::cos(angle), std::sin(angle));
    const Eigen::Vector2d axis_lengths(kEllipseSigma * std::sqrt(std::max(0.0, eigenvalues(0))),
                                        kEllipseSigma * std::sqrt(std::max(0.0, eigenvalues(1))));
    const Eigen::Vector2d local_offset(axis_lengths.x() * unit_circle.x(), axis_lengths.y() * unit_circle.y());
    const Eigen::Vector2d world_point = center + eigenvectors * local_offset;
    points.emplace_back(static_cast<float>(world_point.x() - origin.x()), static_cast<float>(world_point.y() - origin.y()),
                         static_cast<float>(height - origin.z()));
  }
  return points;
}

// Draws one frame's whole scene: the lidar points coloured by height, the ego box, and every
// confirmed track's box, history, prediction and uncertainty ellipse. Any track that was drawn on
// the previous call but is not in this frame's track list gets recursively cleared so its box and
// every one of its sub-entities disappear rather than freezing on screen. This is the one function
// both the live viewer and the saved recording call, so what a person watches live and what ends
// up in the .rrd file are always built the same way.
void log_frame(rerun::RecordingStream& stream, std::size_t frame_index, const Frame& frame,
               const std::vector<Track>& tracks, const std::vector<PredictedPath>& predictions,
               const Eigen::Vector3d& origin, std::vector<std::uint64_t>& previously_logged_track_ids) {
  stream.set_time_sequence("frame", static_cast<std::int64_t>(frame_index));

  const Eigen::Matrix3d rotation = frame.vehicle_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3d translation = frame.vehicle_to_world.block<3, 1>(0, 3);

  std::vector<rerun::Position3D> point_positions;
  std::vector<rerun::Color> point_colors;
  point_positions.reserve(frame.points.size());
  point_colors.reserve(frame.points.size());
  for (const Point& point : frame.points) {
    const Eigen::Vector3d local_position(point.x, point.y, point.z);
    const Eigen::Vector3d world_position = rotation * local_position + translation - origin;
    point_positions.emplace_back(static_cast<float>(world_position.x()), static_cast<float>(world_position.y()),
                                  static_cast<float>(world_position.z()));
    point_colors.push_back(height_ramp_color(static_cast<float>(world_position.z())));
  }
  stream.log("world/points", rerun::Points3D(point_positions)
                                  .with_radii(rerun::Radius(static_cast<float>(kPointRadius)))
                                  .with_colors(point_colors));

  const Eigen::Vector3d ego_translation = translation - origin;
  const double ego_yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  stream.log("world/ego",
             rerun::Boxes3D::from_centers_and_half_sizes(
                 rerun::components::PoseTranslation3D(static_cast<float>(ego_translation.x()),
                                                    static_cast<float>(ego_translation.y()),
                                                    static_cast<float>(ego_translation.z())),
                 rerun::HalfSize3D(2.6f, 1.2f, 0.9f))
                 .with_quaternions(rerun::components::PoseRotationQuat(
                     rerun::Quaternion::from_xyzw(0.0f, 0.0f, static_cast<float>(std::sin(ego_yaw / 2.0)),
                                                   static_cast<float>(std::cos(ego_yaw / 2.0)))))
                 .with_colors(rerun::Color(200, 200, 200)));

  std::vector<std::uint64_t> currently_logged_track_ids;
  currently_logged_track_ids.reserve(tracks.size());
  for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index) {
    const Track& track = tracks[track_index];
    currently_logged_track_ids.push_back(track.track_id);
    const std::string track_path = "world/tracks/" + std::to_string(track.track_id);
    const rerun::Color track_color = class_color(track.object_class);

    const Eigen::Vector2d track_position(track.state.mean(0), track.state.mean(1));
    const double track_yaw = track.state.mean(2);
    const Eigen::Vector3d box_center(track_position.x() - origin.x(), track_position.y() - origin.y(),
                                      track.state.center_z - origin.z());

    stream.log(track_path,
               rerun::Boxes3D::from_centers_and_half_sizes(
                   rerun::components::PoseTranslation3D(static_cast<float>(box_center.x()), static_cast<float>(box_center.y()),
                                                      static_cast<float>(box_center.z())),
                   rerun::HalfSize3D(static_cast<float>(track.state.length / 2.0), static_cast<float>(track.state.width / 2.0),
                                      static_cast<float>(track.state.height / 2.0)))
                   .with_quaternions(rerun::components::PoseRotationQuat(rerun::Quaternion::from_xyzw(
                       0.0f, 0.0f, static_cast<float>(std::sin(track_yaw / 2.0)), static_cast<float>(std::cos(track_yaw / 2.0)))))
                   .with_colors(track_color)
                   .with_labels(rerun::Text(std::to_string(track.track_id).c_str())));

    std::vector<rerun::Vec3D> history_points;
    history_points.reserve(track.history.size());
    for (const Eigen::Vector2d& history_point : track.history)
      history_points.emplace_back(static_cast<float>(history_point.x() - origin.x()),
                                   static_cast<float>(history_point.y() - origin.y()),
                                   static_cast<float>(track.state.center_z - origin.z()));
    stream.log(track_path + "/history",
               rerun::LineStrips3D(rerun::components::LineStrip3D(history_points)).with_colors(track_color));

    const PredictedPath& prediction = predictions[track_index];
    std::vector<rerun::Vec3D> prediction_points;
    prediction_points.reserve(prediction.positions.size());
    for (const Eigen::Vector2d& predicted_position : prediction.positions)
      prediction_points.emplace_back(static_cast<float>(predicted_position.x() - origin.x()),
                                      static_cast<float>(predicted_position.y() - origin.y()),
                                      static_cast<float>(track.state.center_z - origin.z()));
    stream.log(track_path + "/prediction",
               rerun::LineStrips3D(rerun::components::LineStrip3D(prediction_points)).with_colors(track_color));

    if (!prediction.covariances.empty()) {
      const std::vector<rerun::Vec3D> uncertainty_points =
          ellipse_points(prediction.positions.back(), prediction.covariances.back(), track.state.center_z, origin);
      stream.log(track_path + "/uncertainty",
                 rerun::LineStrips3D(rerun::components::LineStrip3D(uncertainty_points)).with_colors(track_color));
    }
  }

  for (std::uint64_t previous_track_id : previously_logged_track_ids) {
    const bool still_present = std::find(currently_logged_track_ids.begin(), currently_logged_track_ids.end(),
                                          previous_track_id) != currently_logged_track_ids.end();
    if (!still_present) stream.log("world/tracks/" + std::to_string(previous_track_id), rerun::Clear::RECURSIVE);
  }
  previously_logged_track_ids = std::move(currently_logged_track_ids);
}

// Logs the one number a class's tracking quality boils down to, three times over, plus how fast
// the tracker itself is running. This is the same TrackingMetrics and TimingStats the headless
// summary prints - the viewer never computes a number of its own, it only draws Waymo's CLEAR
// accounting as it goes.
void log_metrics(rerun::RecordingStream& stream, const std::unordered_map<ObjectClass, ClassMetrics>& metrics,
                  double step_p50_milliseconds, double step_p99_milliseconds, std::uint64_t overruns) {
  for (const auto& [object_class, class_metrics] : metrics) {
    const std::string class_path = "metrics/" + class_name(object_class);
    stream.log(class_path + "/mota", rerun::Scalars(class_metrics.mota()));
    stream.log(class_path + "/motp", rerun::Scalars(class_metrics.motp()));
    stream.log(class_path + "/id_switches", rerun::Scalars(static_cast<double>(class_metrics.id_switches)));
  }
  stream.log("timing/step_p50_ms", rerun::Scalars(step_p50_milliseconds));
  stream.log("timing/step_p99_ms", rerun::Scalars(step_p99_milliseconds));
  stream.log("timing/overruns", rerun::Scalars(static_cast<double>(overruns)));
}

}  // namespace

// Opens the live Rerun stream (when log_to_rerun asks for one) and a small GLFW/ImGui window of
// sliders, then loops until that window closes or controls.quit is set from the replay side.
// Every pass through the loop asks the snapshot exchange for the newest published frame and, if
// it is one this function has not drawn yet, hands it to log_frame and log_metrics; the sliders
// write straight into the same atomics the replay thread reads its perturbation settings from, so
// dragging one takes effect on the very next frame the replay produces. Closing the window sets
// controls.quit so the replay thread the caller started on its own always stops rather than
// running on with nothing left to show for it.
int run_viewer(SnapshotExchange& exchange, LiveControls& controls, const SegmentLog& segment, bool log_to_rerun) {
  rerun::RecordingStream stream("tracker");
  if (log_to_rerun) stream.spawn().exit_on_failure();
  const Eigen::Vector3d origin = recording_origin(segment);

  if (!glfwInit()) return 1;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  GLFWwindow* window = glfwCreateWindow(420, 300, "tracker controls", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 150");

  std::size_t last_logged_frame_index = static_cast<std::size_t>(-1);
  std::vector<std::uint64_t> previously_logged_track_ids;

  while (glfwWindowShouldClose(window) == 0 && !controls.quit.load()) {
    glfwPollEvents();

    const Snapshot* snapshot = exchange.acquire();
    if (log_to_rerun && snapshot->frame != nullptr && snapshot->frame_index != last_logged_frame_index) {
      log_frame(stream, snapshot->frame_index, *snapshot->frame, snapshot->tracks, snapshot->predictions, origin,
                previously_logged_track_ids);
      log_metrics(stream, snapshot->metrics, snapshot->step_p50_milliseconds, snapshot->step_p99_milliseconds,
                  snapshot->overruns);
      last_logged_frame_index = snapshot->frame_index;
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int window_width = 0, window_height = 0;
    glfwGetFramebufferSize(window, &window_width, &window_height);
    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(window_width), static_cast<float>(window_height)));
    ImGui::Begin("tracker controls", nullptr,
                  ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                      ImGuiWindowFlags_NoTitleBar);

    float dropout_probability = static_cast<float>(controls.dropout_probability.load());
    if (ImGui::SliderFloat("dropout", &dropout_probability, 0.0f, 1.0f))
      controls.dropout_probability.store(static_cast<double>(dropout_probability));

    float position_noise_metres = static_cast<float>(controls.position_noise_metres.load());
    if (ImGui::SliderFloat("position noise (m)", &position_noise_metres, 0.0f, 2.0f))
      controls.position_noise_metres.store(static_cast<double>(position_noise_metres));

    float latency_milliseconds = static_cast<float>(controls.latency_micros.load()) / 1000.0f;
    if (ImGui::SliderFloat("latency (ms)", &latency_milliseconds, 0.0f, 500.0f))
      controls.latency_micros.store(static_cast<std::int64_t>(latency_milliseconds * 1000.0f));

    bool paused = controls.paused.load();
    if (ImGui::Checkbox("paused", &paused)) controls.paused.store(paused);

    ImGui::Separator();
    ImGui::Text("frame %zu", snapshot->frame_index);
    for (const auto& [object_class, class_metrics] : snapshot->metrics) {
      ImGui::Text("%s mota %.3f motp %.3f id_switches %llu", class_name(object_class).c_str(), class_metrics.mota(),
                  class_metrics.motp(), static_cast<unsigned long long>(class_metrics.id_switches));
    }
    ImGui::Text("step p50 %.2f ms p99 %.2f ms overruns %llu", snapshot->step_p50_milliseconds,
                snapshot->step_p99_milliseconds, static_cast<unsigned long long>(snapshot->overruns));

    ImGui::End();
    ImGui::Render();

    glViewport(0, 0, window_width, window_height);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  controls.quit = true;
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}

// Writes a finished replay's stored tracks back out through log_frame into a saved .rrd file,
// recomputing each frame's predictions from the same predictor the live run used. Nothing here
// runs live and nothing paces itself against a clock, so a segment that took minutes to replay in
// real time is written out as fast as the predictor and the Rerun SDK can manage; the result is a
// gapless recording of exactly what the viewer would have shown, for later playback with no
// tracker running at all.
void save_replay_recording(const std::string& path, const SegmentLog& segment,
                            const std::vector<std::vector<Track>>& confirmed_tracks_per_frame,
                            const Predictor& predictor, double horizon_seconds, double step_seconds) {
  rerun::RecordingStream stream("tracker");
  stream.save(path).exit_on_failure();
  const Eigen::Vector3d origin = recording_origin(segment);
  std::vector<std::uint64_t> previously_logged_track_ids;

  const std::size_t frame_count = std::min(segment.frames.size(), confirmed_tracks_per_frame.size());
  for (std::size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    const std::vector<Track>& tracks = confirmed_tracks_per_frame[frame_index];
    std::vector<PredictedPath> predictions;
    predictions.reserve(tracks.size());
    for (const Track& track : tracks) predictions.push_back(predictor.predict(track, horizon_seconds, step_seconds));
    log_frame(stream, frame_index, segment.frames[frame_index], tracks, predictions, origin, previously_logged_track_ids);
  }
}
