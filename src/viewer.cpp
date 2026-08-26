#include "viewer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <foxglove/mcap.hpp>
#include <foxglove/messages.hpp>

// This file is the only place the project draws anything. save_replay_recording writes the lidar
// points, the ego box, and every confirmed track's box, history trail, predicted path and
// uncertainty ellipse into an MCAP file, using the Foxglove SDK's well-known schemas, for later
// playback in Lichtblick. A small ImGui window still runs live, next to whatever tracker is
// running - it lets the perturbation sliders move the atomics in LiveControls and prints the same
// numbers the headless summary prints, but it draws nothing of the scene itself; the recording is
// the only place a person watching sees the lidar points and the tracks. Every position written
// out is first shifted by recording_origin's translation, because a raw Waymo world coordinate can
// be tens of thousands of metres from zero and would burn through a float's precision long before
// it reached the file; write_frame is the one function the recorder calls once per frame, so the
// saved file is always built the same way regardless of which run produced it.

namespace {

constexpr double kPointHeightLow = -2.0;
constexpr double kPointHeightHigh = 4.0;
constexpr int kEllipseSegments = 32;
constexpr double kEllipseSigma = 2.0;
constexpr double kLineThickness = 0.08;
constexpr double kLabelFontSize = 0.6;
constexpr double kEgoLength = 5.2, kEgoWidth = 2.4, kEgoHeight = 1.8;

// The one colour a class is ever drawn in, everywhere a track of that class appears: its box, its
// history trail, its predicted path, its uncertainty ellipse and its label all share this so an
// object reads as the same object across every primitive it is drawn with.
foxglove::messages::Color class_color(ObjectClass object_class) {
  switch (object_class) {
    case ObjectClass::Vehicle: return foxglove::messages::Color{0.0, 0.59, 1.0, 1.0};
    case ObjectClass::Pedestrian: return foxglove::messages::Color{1.0, 0.55, 0.0, 1.0};
    case ObjectClass::Cyclist: return foxglove::messages::Color{0.0, 0.86, 0.47, 1.0};
  }
  return foxglove::messages::Color{1.0, 1.0, 1.0, 1.0};
}

// The name a class is filed under in the controls window's metrics text. Nothing about tracking
// uses this string; it only exists so a human reading the window can tell the three classes apart.
std::string class_name(ObjectClass object_class) {
  switch (object_class) {
    case ObjectClass::Vehicle: return "vehicle";
    case ObjectClass::Pedestrian: return "pedestrian";
    case ObjectClass::Cyclist: return "cyclist";
  }
  return "unknown";
}

// The translation of frame 0's vehicle pose, in Waymo's world frame. Every position this file ever
// writes out has this subtracted from it first, because a segment's world coordinates can sit tens
// of thousands of metres from the origin and a float only has so many bits of precision to spend -
// without this shift, a lidar point and a track box a few metres apart could round to the same
// float and jitter in the recording.
Eigen::Vector3d recording_origin(const SegmentLog& segment) {
  return segment.frames.front().vehicle_to_world.block<3, 1>(0, 3);
}

// Turns a frame's capture time into the seconds-and-nanoseconds pair the Foxglove schemas want on
// a message, and into the single nanosecond count the MCAP channel wants as a message's log time.
// Both come from the same field so a message's timestamp and the position it sits at in the file
// always agree.
foxglove::messages::Timestamp to_timestamp(std::int64_t capture_time_micros) {
  return foxglove::messages::Timestamp{static_cast<std::uint32_t>(capture_time_micros / 1'000'000),
                                        static_cast<std::uint32_t>((capture_time_micros % 1'000'000) * 1'000)};
}

std::uint64_t to_nanoseconds(std::int64_t capture_time_micros) {
  return static_cast<std::uint64_t>(capture_time_micros) * 1'000;
}

// A pose with no translation and no rotation, used for the point cloud, whose points already carry
// their own world position in the packed data rather than being offset by the channel's pose.
foxglove::messages::Pose identity_pose() {
  foxglove::messages::Pose pose;
  pose.position = foxglove::messages::Vector3{0.0, 0.0, 0.0};
  pose.orientation = foxglove::messages::Quaternion{0.0, 0.0, 0.0, 1.0};
  return pose;
}

// The pose every box in the scene is drawn with: a position and a rotation about the z axis built
// from a yaw angle, which is all the tracker ever estimates - nothing in this project models roll
// or pitch.
foxglove::messages::Pose pose_from_translation_yaw(const Eigen::Vector3d& translation, double yaw) {
  foxglove::messages::Pose pose;
  pose.position = foxglove::messages::Vector3{translation.x(), translation.y(), translation.z()};
  pose.orientation = foxglove::messages::Quaternion{0.0, 0.0, std::sin(yaw / 2.0), std::cos(yaw / 2.0)};
  return pose;
}

// Appends the raw bytes of one value onto a point cloud's packed data buffer. This is the only way
// the PointCloud schema carries per-point data - every point's x, y, z and colour are laid down
// back to back at the stride write_frame declares, and this is the one place that byte layout is
// built.
template <typename Value>
void append_bytes(std::vector<std::byte>& buffer, const Value& value) {
  const auto* value_bytes = reinterpret_cast<const std::byte*>(&value);
  buffer.insert(buffer.end(), value_bytes, value_bytes + sizeof(Value));
}

// Turns a height above the ground into the packed colour Lichtblick's point cloud renderer expects
// in an "rgba" field: a little-endian uint32 read back as 0xaarrggbb, which means blue occupies the
// lowest byte and alpha the highest. The height is clamped to [kPointHeightLow, kPointHeightHigh]
// first so a stray high or low point cannot pull the ramp's range around for every other point in
// the same cloud, and the ramp itself runs blue to green to yellow as height increases.
std::uint32_t height_ramp_color(float height) {
  const float clamped_height = std::clamp(height, static_cast<float>(kPointHeightLow), static_cast<float>(kPointHeightHigh));
  const float ramp_fraction = (clamped_height - static_cast<float>(kPointHeightLow)) /
                               static_cast<float>(kPointHeightHigh - kPointHeightLow);

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

  const std::uint32_t red = static_cast<std::uint32_t>(ramp_color.x() * 255.0f);
  const std::uint32_t green_channel = static_cast<std::uint32_t>(ramp_color.y() * 255.0f);
  const std::uint32_t blue_channel = static_cast<std::uint32_t>(ramp_color.z() * 255.0f);
  const std::uint32_t alpha = 255;
  return (alpha << 24) | (red << 16) | (green_channel << 8) | blue_channel;
}

// Builds the closed ring of points that draws as an uncertainty ellipse: kEllipseSigma standard
// deviations along each eigenvector of the given 2x2 covariance, centred on the given position, at
// the given height. This is the shape the recording draws around a track's final predicted
// position, so a wide ellipse there means the filter has stopped trusting its own guess. The ring
// is logged as a closed LINE_LOOP, so the points stop one short of the starting angle rather than
// repeating it.
std::vector<foxglove::messages::Point3> ellipse_points(const Eigen::Vector2d& center, const Eigen::Matrix2d& covariance,
                                                         double height, const Eigen::Vector3d& origin) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(covariance);
  const Eigen::Vector2d eigenvalues = eigen_solver.eigenvalues();
  const Eigen::Matrix2d eigenvectors = eigen_solver.eigenvectors();

  std::vector<foxglove::messages::Point3> points;
  points.reserve(kEllipseSegments);
  for (int segment_index = 0; segment_index < kEllipseSegments; ++segment_index) {
    const double angle = 2.0 * M_PI * static_cast<double>(segment_index) / static_cast<double>(kEllipseSegments);
    const Eigen::Vector2d unit_circle(std::cos(angle), std::sin(angle));
    const Eigen::Vector2d axis_lengths(kEllipseSigma * std::sqrt(std::max(0.0, eigenvalues(0))),
                                        kEllipseSigma * std::sqrt(std::max(0.0, eigenvalues(1))));
    const Eigen::Vector2d local_offset(axis_lengths.x() * unit_circle.x(), axis_lengths.y() * unit_circle.y());
    const Eigen::Vector2d world_point = center + eigenvectors * local_offset;
    points.push_back(foxglove::messages::Point3{world_point.x() - origin.x(), world_point.y() - origin.y(), height - origin.z()});
  }
  return points;
}

// Writes one frame's whole scene to the three channels: the transform that places the ego frame
// inside the world frame, the lidar points coloured by height, the ego box, and every confirmed
// track's box, history, prediction and uncertainty ellipse. Without this transform, "world" is a
// frame name every other message refers to but nothing ever establishes, so Lichtblick's 3D panel
// has no display frame to offer and renders nothing. Any track that was written on the previous
// call but is not in this frame's track list is added to this frame's deletions so its box and
// every one of its primitives disappear from the recording rather than freezing in place. This is
// the one function both save_replay_recording and, were it ever run live, a live sink would call,
// so a recording is always built the same way regardless of who asked for it.
void write_frame(foxglove::messages::FrameTransformChannel& transform_channel, foxglove::messages::PointCloudChannel& points_channel,
                  foxglove::messages::SceneUpdateChannel& scene_channel, const Frame& frame, const std::vector<Track>& tracks,
                  const std::vector<PredictedPath>& predictions, const Eigen::Vector3d& origin,
                  std::vector<std::uint64_t>& previously_logged_track_ids) {
  const std::uint64_t log_time = to_nanoseconds(frame.capture_time_micros);
  const foxglove::messages::Timestamp timestamp = to_timestamp(frame.capture_time_micros);

  const Eigen::Matrix3d rotation = frame.vehicle_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3d translation = frame.vehicle_to_world.block<3, 1>(0, 3);
  const Eigen::Vector3d ego_translation = translation - origin;
  const double ego_yaw = std::atan2(rotation(1, 0), rotation(0, 0));

  foxglove::messages::FrameTransform frame_transform;
  frame_transform.timestamp = timestamp;
  frame_transform.parent_frame_id = "world";
  frame_transform.child_frame_id = "ego";
  frame_transform.translation = foxglove::messages::Vector3{ego_translation.x(), ego_translation.y(), ego_translation.z()};
  frame_transform.rotation =
      foxglove::messages::Quaternion{0.0, 0.0, std::sin(ego_yaw / 2.0), std::cos(ego_yaw / 2.0)};
  const foxglove::FoxgloveError transform_error = transform_channel.log(frame_transform, log_time);
  if (transform_error != foxglove::FoxgloveError::Ok)
    throw std::runtime_error(std::string("cannot log frame transform: ") + foxglove::strerror(transform_error));

  foxglove::messages::PointCloud point_cloud;
  point_cloud.timestamp = timestamp;
  point_cloud.frame_id = "world";
  point_cloud.pose = identity_pose();
  point_cloud.point_stride = 16;
  point_cloud.fields = {
      foxglove::messages::PackedElementField{"x", 0, foxglove::messages::PackedElementField::NumericType::FLOAT32},
      foxglove::messages::PackedElementField{"y", 4, foxglove::messages::PackedElementField::NumericType::FLOAT32},
      foxglove::messages::PackedElementField{"z", 8, foxglove::messages::PackedElementField::NumericType::FLOAT32},
      foxglove::messages::PackedElementField{"rgba", 12, foxglove::messages::PackedElementField::NumericType::UINT32},
  };
  point_cloud.data.reserve(frame.points.size() * point_cloud.point_stride);
  for (const Point& point : frame.points) {
    const Eigen::Vector3d local_position(point.x, point.y, point.z);
    const Eigen::Vector3d world_position = rotation * local_position + translation - origin;
    append_bytes(point_cloud.data, static_cast<float>(world_position.x()));
    append_bytes(point_cloud.data, static_cast<float>(world_position.y()));
    append_bytes(point_cloud.data, static_cast<float>(world_position.z()));
    append_bytes(point_cloud.data, height_ramp_color(static_cast<float>(world_position.z())));
  }
  const foxglove::FoxgloveError points_error = points_channel.log(point_cloud, log_time);
  if (points_error != foxglove::FoxgloveError::Ok)
    throw std::runtime_error(std::string("cannot log point cloud: ") + foxglove::strerror(points_error));

  foxglove::messages::SceneUpdate scene_update;

  foxglove::messages::SceneEntity ego_entity;
  ego_entity.timestamp = timestamp;
  ego_entity.frame_id = "world";
  ego_entity.id = "ego";
  foxglove::messages::CubePrimitive ego_cube;
  ego_cube.pose = pose_from_translation_yaw(ego_translation, ego_yaw);
  ego_cube.size = foxglove::messages::Vector3{kEgoLength, kEgoWidth, kEgoHeight};
  ego_cube.color = foxglove::messages::Color{0.8, 0.8, 0.8, 1.0};
  ego_entity.cubes.push_back(std::move(ego_cube));
  scene_update.entities.push_back(std::move(ego_entity));

  std::vector<std::uint64_t> currently_logged_track_ids;
  currently_logged_track_ids.reserve(tracks.size());
  for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index) {
    const Track& track = tracks[track_index];
    currently_logged_track_ids.push_back(track.track_id);
    const foxglove::messages::Color track_color = class_color(track.object_class);

    const Eigen::Vector2d track_position(track.state.mean(0), track.state.mean(1));
    const double track_yaw = track.state.mean(2);
    const Eigen::Vector3d box_center(track_position.x() - origin.x(), track_position.y() - origin.y(),
                                      track.state.center_z - origin.z());

    foxglove::messages::SceneEntity track_entity;
    track_entity.timestamp = timestamp;
    track_entity.frame_id = "world";
    track_entity.id = std::to_string(track.track_id);

    foxglove::messages::CubePrimitive track_cube;
    track_cube.pose = pose_from_translation_yaw(box_center, track_yaw);
    track_cube.size = foxglove::messages::Vector3{track.state.length, track.state.width, track.state.height};
    track_cube.color = foxglove::messages::Color{track_color.r, track_color.g, track_color.b, 0.35};
    track_entity.cubes.push_back(std::move(track_cube));

    foxglove::messages::LinePrimitive history_line;
    history_line.type = foxglove::messages::LinePrimitive::LineType::LINE_STRIP;
    history_line.thickness = kLineThickness;
    history_line.color = track_color;
    history_line.points.reserve(track.history.size());
    for (const Eigen::Vector2d& history_point : track.history)
      history_line.points.push_back(foxglove::messages::Point3{history_point.x() - origin.x(), history_point.y() - origin.y(),
                                                                 track.state.center_z - origin.z()});
    track_entity.lines.push_back(std::move(history_line));

    const PredictedPath& prediction = predictions[track_index];
    foxglove::messages::LinePrimitive prediction_line;
    prediction_line.type = foxglove::messages::LinePrimitive::LineType::LINE_STRIP;
    prediction_line.thickness = kLineThickness;
    prediction_line.color = track_color;
    prediction_line.points.reserve(prediction.positions.size());
    for (const Eigen::Vector2d& predicted_position : prediction.positions)
      prediction_line.points.push_back(foxglove::messages::Point3{predicted_position.x() - origin.x(),
                                                                    predicted_position.y() - origin.y(),
                                                                    track.state.center_z - origin.z()});
    track_entity.lines.push_back(std::move(prediction_line));

    if (!prediction.covariances.empty()) {
      foxglove::messages::LinePrimitive uncertainty_line;
      uncertainty_line.type = foxglove::messages::LinePrimitive::LineType::LINE_LOOP;
      uncertainty_line.thickness = kLineThickness;
      uncertainty_line.color = track_color;
      uncertainty_line.points = ellipse_points(prediction.positions.back(), prediction.covariances.back(), track.state.center_z, origin);
      track_entity.lines.push_back(std::move(uncertainty_line));
    }

    foxglove::messages::TextPrimitive label;
    label.pose = pose_from_translation_yaw(box_center + Eigen::Vector3d(0.0, 0.0, track.state.height / 2.0), 0.0);
    label.billboard = true;
    label.font_size = kLabelFontSize;
    label.scale_invariant = true;
    label.color = track_color;
    label.text = std::to_string(track.track_id);
    track_entity.texts.push_back(std::move(label));

    scene_update.entities.push_back(std::move(track_entity));
  }

  for (std::uint64_t previous_track_id : previously_logged_track_ids) {
    const bool still_present = std::find(currently_logged_track_ids.begin(), currently_logged_track_ids.end(),
                                          previous_track_id) != currently_logged_track_ids.end();
    if (still_present) continue;
    foxglove::messages::SceneEntityDeletion deletion;
    deletion.timestamp = timestamp;
    deletion.type = foxglove::messages::SceneEntityDeletion::SceneEntityDeletionType::MATCHING_ID;
    deletion.id = std::to_string(previous_track_id);
    scene_update.deletions.push_back(std::move(deletion));
  }
  previously_logged_track_ids = std::move(currently_logged_track_ids);

  const foxglove::FoxgloveError scene_error = scene_channel.log(scene_update, log_time);
  if (scene_error != foxglove::FoxgloveError::Ok)
    throw std::runtime_error(std::string("cannot log scene update: ") + foxglove::strerror(scene_error));
}

}  // namespace

// Opens a small GLFW/ImGui window of perturbation sliders and loops until that window closes or
// controls.quit is set from the replay side. Every pass through the loop asks the snapshot
// exchange for the newest published frame so the printed metrics and timing text stay current; the
// sliders write straight into the same atomics the replay thread reads its perturbation settings
// from, so dragging one takes effect on the very next frame the replay produces. This window draws
// no part of the scene itself - that only ever happens in the recording save_replay_recording
// writes out, so what a person watches live is the sliders and the numbers, not the tracks.
// Closing the window sets controls.quit so the replay thread the caller started on its own always
// stops rather than running on with nothing left to show for it.
int run_viewer(SnapshotExchange& exchange, LiveControls& controls) {
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

  while (glfwWindowShouldClose(window) == 0 && !controls.quit.load()) {
    glfwPollEvents();

    const Snapshot* snapshot = exchange.acquire();

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

// Writes a finished replay's stored tracks out through write_frame into a saved .mcap file,
// recomputing each frame's predictions from the same predictor the live run used. Nothing here
// runs live and nothing paces itself against a clock, so a segment that took minutes to replay in
// real time is written out as fast as the predictor and the Foxglove SDK can manage; the result is
// a gapless recording of exactly what a live viewer would have shown, for later playback in
// Lichtblick with no tracker running at all.
void save_replay_recording(const std::string& path, const SegmentLog& segment,
                            const std::vector<std::vector<Track>>& confirmed_tracks_per_frame,
                            const Predictor& predictor, double horizon_seconds, double step_seconds) {
  foxglove::McapWriterOptions writer_options;
  writer_options.path = path;
  writer_options.compression = foxglove::McapCompression::Zstd;
  foxglove::FoxgloveResult<foxglove::McapWriter> writer_result = foxglove::McapWriter::create(writer_options);
  if (!writer_result.has_value())
    throw std::runtime_error(std::string("cannot create ") + path + ": " + foxglove::strerror(writer_result.error()));
  foxglove::McapWriter writer = std::move(writer_result.value());

  foxglove::FoxgloveResult<foxglove::messages::PointCloudChannel> points_channel_result =
      foxglove::messages::PointCloudChannel::create("/points");
  if (!points_channel_result.has_value())
    throw std::runtime_error(std::string("cannot create /points channel: ") + foxglove::strerror(points_channel_result.error()));
  foxglove::messages::PointCloudChannel points_channel = std::move(points_channel_result.value());

  foxglove::FoxgloveResult<foxglove::messages::FrameTransformChannel> transform_channel_result =
      foxglove::messages::FrameTransformChannel::create("/tf");
  if (!transform_channel_result.has_value())
    throw std::runtime_error(std::string("cannot create /tf channel: ") + foxglove::strerror(transform_channel_result.error()));
  foxglove::messages::FrameTransformChannel transform_channel = std::move(transform_channel_result.value());

  foxglove::FoxgloveResult<foxglove::messages::SceneUpdateChannel> scene_channel_result =
      foxglove::messages::SceneUpdateChannel::create("/scene");
  if (!scene_channel_result.has_value())
    throw std::runtime_error(std::string("cannot create /scene channel: ") + foxglove::strerror(scene_channel_result.error()));
  foxglove::messages::SceneUpdateChannel scene_channel = std::move(scene_channel_result.value());

  const Eigen::Vector3d origin = recording_origin(segment);
  std::vector<std::uint64_t> previously_logged_track_ids;

  const std::size_t frame_count = std::min(segment.frames.size(), confirmed_tracks_per_frame.size());
  for (std::size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    const std::vector<Track>& tracks = confirmed_tracks_per_frame[frame_index];
    std::vector<PredictedPath> predictions;
    predictions.reserve(tracks.size());
    for (const Track& track : tracks) predictions.push_back(predictor.predict(track, horizon_seconds, step_seconds));
    write_frame(transform_channel, points_channel, scene_channel, segment.frames[frame_index], tracks, predictions, origin,
                previously_logged_track_ids);
  }

  const foxglove::FoxgloveError close_error = writer.close();
  if (close_error != foxglove::FoxgloveError::Ok)
    throw std::runtime_error(std::string("cannot close ") + path + ": " + foxglove::strerror(close_error));
}
