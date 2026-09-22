// Writes a replay's lidar points, boxes, trails, predictions and
// uncertainty ellipses into an MCAP file for Lichtblick playback.

#include "recording.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <foxglove/mcap.hpp>
#include <foxglove/messages.hpp>

namespace {

// Drawing constants: point height ramp, ellipse sigma and segments,
// line, label and ego box sizes. Lengths in metres. Chosen.
constexpr double kPointHeightLow = -2.0;
constexpr double kPointHeightHigh = 4.0;
constexpr int kEllipseSegments = 32;
constexpr double kEllipseSigma = 2.0;
constexpr double kLineThickness = 0.08;
constexpr double kLabelFontSize = 0.6;
constexpr double kEgoLength = 5.2, kEgoWidth = 2.4, kEgoHeight = 1.8;
constexpr double kBoxAlpha = 0.35;
constexpr double kEgoGrey = 0.8;

// The one colour a class is drawn in across each primitive, box,
// trail, prediction, ellipse and label, so it reads as one object.
foxglove::messages::Color class_color(ObjectClass object_class) {
  switch (object_class) {
    case ObjectClass::Vehicle:
      return foxglove::messages::Color{0.0, 0.59, 1.0, 1.0};
    case ObjectClass::Pedestrian:
      return foxglove::messages::Color{1.0, 0.55, 0.0, 1.0};
    case ObjectClass::Cyclist:
      return foxglove::messages::Color{0.0, 0.86, 0.47, 1.0};
  }
  return foxglove::messages::Color{1.0, 1.0, 1.0, 1.0};
}

// The world-frame translation of frame 0's pose; each position
// written out has this subtracted to keep it within float precision.
Eigen::Vector3d recording_origin(const SegmentLog& segment) {
  return segment.frames.front().vehicle_to_world.block<3, 1>(0, 3);
}

// Nanoseconds in one microsecond, for converting capture times into
// the units MCAP and its Foxglove timestamps log at.
constexpr std::int64_t kNanosPerMicro = 1'000;

// Converts a capture time in microseconds to the Foxglove
// seconds/nanoseconds timestamp carried on each logged message.
foxglove::messages::Timestamp to_timestamp(std::int64_t capture_time_micros) {
  return foxglove::messages::Timestamp{
      static_cast<std::uint32_t>(capture_time_micros / kMicrosPerSecond),
      static_cast<std::uint32_t>((capture_time_micros % kMicrosPerSecond) *
                                 kNanosPerMicro)};
}

// Converts a capture time in microseconds to nanoseconds, the unit
// an MCAP channel logs a message's time at.
std::uint64_t to_nanoseconds(std::int64_t capture_time_micros) {
  return static_cast<std::uint64_t>(capture_time_micros) * kNanosPerMicro;
}

// A zero position/identity rotation pose, for the point cloud whose
// points already carry their own world position in the packed data.
foxglove::messages::Pose identity_pose() {
  foxglove::messages::Pose pose;
  pose.position = foxglove::messages::Vector3{0.0, 0.0, 0.0};
  pose.orientation = foxglove::messages::Quaternion{0.0, 0.0, 0.0, 1.0};
  return pose;
}

// A pose built from a translation and a yaw about z; boxes never
// need roll or pitch, which the tracker doesn't estimate.
foxglove::messages::Pose pose_from_translation_yaw(
    const Eigen::Vector3d& translation, double yaw) {
  foxglove::messages::Pose pose;
  pose.position = foxglove::messages::Vector3{translation.x(), translation.y(),
                                              translation.z()};
  pose.orientation = foxglove::messages::Quaternion{
      0.0, 0.0, std::sin(yaw / 2.0), std::cos(yaw / 2.0)};
  return pose;
}

// Appends one value's raw bytes onto a point cloud's packed data
// buffer; this is the only way per-point fields get laid down.
template <typename Value>
void append_bytes(std::vector<std::byte>& buffer, const Value& value) {
  const auto* value_bytes = reinterpret_cast<const std::byte*>(&value);
  buffer.insert(buffer.end(), value_bytes, value_bytes + sizeof(Value));
}

// Maps a height to Lichtblick's packed "rgba" uint32 (0xaarrggbb),
// ramping blue to green to yellow after clamping to the drawn range.
std::uint32_t height_ramp_color(float height) {
  const float clamped_height =
      std::clamp(height, static_cast<float>(kPointHeightLow),
                 static_cast<float>(kPointHeightHigh));
  const float ramp_fraction =
      (clamped_height - static_cast<float>(kPointHeightLow)) /
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
  const std::uint32_t green_channel =
      static_cast<std::uint32_t>(ramp_color.y() * 255.0f);
  const std::uint32_t blue_channel =
      static_cast<std::uint32_t>(ramp_color.z() * 255.0f);
  const std::uint32_t alpha = 255;
  return (alpha << 24) | (red << 16) | (green_channel << 8) | blue_channel;
}

// Builds a ring of points kEllipseSigma std devs along each
// covariance eigenvector, for a LINE_LOOP uncertainty ellipse.
std::vector<foxglove::messages::Point3> ellipse_points(
    const Eigen::Vector2d& center, const Eigen::Matrix2d& covariance,
    double height, const Eigen::Vector3d& origin) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(covariance);
  const Eigen::Vector2d eigenvalues = eigen_solver.eigenvalues();
  const Eigen::Matrix2d eigenvectors = eigen_solver.eigenvectors();

  std::vector<foxglove::messages::Point3> points;
  points.reserve(kEllipseSegments);
  // LINE_LOOP closes the ring itself; no repeated last point.
  for (int segment_index = 0; segment_index < kEllipseSegments;
       ++segment_index) {
    const double angle = 2.0 * M_PI * static_cast<double>(segment_index) /
                         static_cast<double>(kEllipseSegments);
    const Eigen::Vector2d unit_circle(std::cos(angle), std::sin(angle));
    const Eigen::Vector2d axis_lengths(
        kEllipseSigma * std::sqrt(std::max(0.0, eigenvalues(0))),
        kEllipseSigma * std::sqrt(std::max(0.0, eigenvalues(1))));
    const Eigen::Vector2d local_offset(axis_lengths.x() * unit_circle.x(),
                                       axis_lengths.y() * unit_circle.y());
    const Eigen::Vector2d world_point = center + eigenvectors * local_offset;
    points.push_back(foxglove::messages::Point3{world_point.x() - origin.x(),
                                                world_point.y() - origin.y(),
                                                height - origin.z()});
  }
  return points;
}

// Turns a Foxglove channel error into a thrown exception, so each
// logging call site fails the same way instead of repeating a check.
void throw_on_error(foxglove::FoxgloveError error, const std::string& action) {
  if (error != foxglove::FoxgloveError::Ok)
    throw std::runtime_error(action + ": " + foxglove::strerror(error));
}

// Creates a Foxglove message channel on the given topic, throwing
// with the channel's own error text if creation fails.
template <typename Channel>
Channel create_channel(const std::string& topic) {
  foxglove::FoxgloveResult<Channel> result = Channel::create(topic);
  if (!result.has_value())
    throw std::runtime_error("cannot create " + topic +
                             " channel: " + foxglove::strerror(result.error()));
  return std::move(result.value());
}

// The three channels a recording writes to; member order fixes the
// channel ids Foxglove assigns them in the file.
struct RecordingChannels {
  foxglove::messages::PointCloudChannel points;
  foxglove::messages::FrameTransformChannel transform;
  foxglove::messages::SceneUpdateChannel scene;
};

// Builds the ego-to-world transform logged each frame, in the
// recording's origin-shifted world frame.
foxglove::messages::FrameTransform build_frame_transform(
    const foxglove::messages::Timestamp& timestamp,
    const Eigen::Vector3d& ego_translation, double ego_yaw) {
  foxglove::messages::FrameTransform frame_transform;
  frame_transform.timestamp = timestamp;
  frame_transform.parent_frame_id = "world";
  frame_transform.child_frame_id = "ego";
  frame_transform.translation = foxglove::messages::Vector3{
      ego_translation.x(), ego_translation.y(), ego_translation.z()};
  frame_transform.rotation = foxglove::messages::Quaternion{
      0.0, 0.0, std::sin(ego_yaw / 2.0), std::cos(ego_yaw / 2.0)};
  return frame_transform;
}

// Bytes per packed point: x, y, z as float32, then rgba as uint32.
constexpr std::uint32_t kPointStrideBytes = 16;

// Builds one frame's lidar point cloud, packing each point's
// origin-shifted world position and height-ramp colour into bytes.
foxglove::messages::PointCloud build_point_cloud(
    const Frame& frame, const foxglove::messages::Timestamp& timestamp,
    const Eigen::Vector3d& origin) {
  const Eigen::Matrix3d rotation = frame.vehicle_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3d translation = frame.vehicle_to_world.block<3, 1>(0, 3);

  foxglove::messages::PointCloud point_cloud;
  point_cloud.timestamp = timestamp;
  point_cloud.frame_id = "world";
  point_cloud.pose = identity_pose();
  point_cloud.point_stride = kPointStrideBytes;
  // Offsets must match the order append_bytes packs bytes below.
  point_cloud.fields = {
      foxglove::messages::PackedElementField{
          "x", 0, foxglove::messages::PackedElementField::NumericType::FLOAT32},
      foxglove::messages::PackedElementField{
          "y", 4, foxglove::messages::PackedElementField::NumericType::FLOAT32},
      foxglove::messages::PackedElementField{
          "z", 8, foxglove::messages::PackedElementField::NumericType::FLOAT32},
      foxglove::messages::PackedElementField{
          "rgba", 12,
          foxglove::messages::PackedElementField::NumericType::UINT32},
  };
  point_cloud.data.reserve(frame.points.size() * point_cloud.point_stride);
  for (const Point& point : frame.points) {
    const Eigen::Vector3d local_position(point.x, point.y, point.z);
    const Eigen::Vector3d world_position =
        rotation * local_position + translation - origin;
    append_bytes(point_cloud.data, static_cast<float>(world_position.x()));
    append_bytes(point_cloud.data, static_cast<float>(world_position.y()));
    append_bytes(point_cloud.data, static_cast<float>(world_position.z()));
    append_bytes(point_cloud.data,
                 height_ramp_color(static_cast<float>(world_position.z())));
  }
  return point_cloud;
}

// Builds the ego vehicle's scene entity: a single grey cube at its
// origin-shifted pose.
foxglove::messages::SceneEntity build_ego_entity(
    const foxglove::messages::Timestamp& timestamp,
    const Eigen::Vector3d& ego_translation, double ego_yaw) {
  foxglove::messages::SceneEntity ego_entity;
  ego_entity.timestamp = timestamp;
  ego_entity.frame_id = "world";
  ego_entity.id = "ego";
  foxglove::messages::CubePrimitive ego_cube;
  ego_cube.pose = pose_from_translation_yaw(ego_translation, ego_yaw);
  ego_cube.size =
      foxglove::messages::Vector3{kEgoLength, kEgoWidth, kEgoHeight};
  ego_cube.color = foxglove::messages::Color{kEgoGrey, kEgoGrey, kEgoGrey, 1.0};
  ego_entity.cubes.push_back(std::move(ego_cube));
  return ego_entity;
}

// Flattens 2D positions to origin-shifted Point3 at a fixed height,
// shared by a track's history trail and its predicted path.
std::vector<foxglove::messages::Point3> flat_path_points(
    const std::vector<Eigen::Vector2d>& positions, double height,
    const Eigen::Vector3d& origin) {
  std::vector<foxglove::messages::Point3> points;
  points.reserve(positions.size());
  for (const Eigen::Vector2d& position : positions)
    points.push_back(foxglove::messages::Point3{position.x() - origin.x(),
                                                position.y() - origin.y(),
                                                height - origin.z()});
  return points;
}

// Builds a line primitive of the given type and colour from already
// origin-shifted points; shared by history, prediction and ellipse.
foxglove::messages::LinePrimitive build_line(
    foxglove::messages::LinePrimitive::LineType type,
    const foxglove::messages::Color& color,
    std::vector<foxglove::messages::Point3> points) {
  foxglove::messages::LinePrimitive line;
  line.type = type;
  line.thickness = kLineThickness;
  line.color = color;
  line.points = std::move(points);
  return line;
}

// Builds a track's id label, billboarded above the box centre so it
// always faces the viewer.
foxglove::messages::TextPrimitive build_label(
    const Track& track, const Eigen::Vector3d& box_center,
    const foxglove::messages::Color& color) {
  foxglove::messages::TextPrimitive label;
  label.pose = pose_from_translation_yaw(
      box_center + Eigen::Vector3d(0.0, 0.0, track.state.height / 2.0), 0.0);
  label.billboard = true;
  label.font_size = kLabelFontSize;
  label.scale_invariant = true;
  label.color = color;
  label.text = std::to_string(track.track_id);
  return label;
}

// Builds one track's scene entity: box, history and prediction
// lines, an optional uncertainty ellipse, and its id label.
foxglove::messages::SceneEntity build_track_entity(
    const Track& track, const PredictedPath& prediction,
    const foxglove::messages::Timestamp& timestamp,
    const Eigen::Vector3d& origin) {
  const foxglove::messages::Color track_color = class_color(track.object_class);

  const Eigen::Vector2d track_position(track.state.mean(kPositionXIndex),
                                       track.state.mean(kPositionYIndex));
  const double track_yaw = track.state.mean(kYawIndex);
  const Eigen::Vector3d box_center(track_position.x() - origin.x(),
                                   track_position.y() - origin.y(),
                                   track.state.center_z - origin.z());

  foxglove::messages::SceneEntity track_entity;
  track_entity.timestamp = timestamp;
  track_entity.frame_id = "world";
  track_entity.id = std::to_string(track.track_id);

  foxglove::messages::CubePrimitive track_cube;
  track_cube.pose = pose_from_translation_yaw(box_center, track_yaw);
  track_cube.size = foxglove::messages::Vector3{
      track.state.length, track.state.width, track.state.height};
  track_cube.color = foxglove::messages::Color{track_color.r, track_color.g,
                                               track_color.b, kBoxAlpha};
  track_entity.cubes.push_back(std::move(track_cube));

  track_entity.lines.push_back(build_line(
      foxglove::messages::LinePrimitive::LineType::LINE_STRIP, track_color,
      flat_path_points(track.history, track.state.center_z, origin)));

  track_entity.lines.push_back(build_line(
      foxglove::messages::LinePrimitive::LineType::LINE_STRIP, track_color,
      flat_path_points(prediction.positions, track.state.center_z, origin)));

  if (!prediction.covariances.empty())
    track_entity.lines.push_back(build_line(
        foxglove::messages::LinePrimitive::LineType::LINE_LOOP, track_color,
        ellipse_points(prediction.positions.back(),
                       prediction.covariances.back(), track.state.center_z,
                       origin)));

  track_entity.texts.push_back(build_label(track, box_center, track_color));

  return track_entity;
}

// Builds one deletion per track present in the previous frame but
// absent from this one, so Lichtblick removes its stale entity.
std::vector<foxglove::messages::SceneEntityDeletion> build_deletions(
    const std::vector<std::uint64_t>& previously_logged_track_ids,
    const std::vector<std::uint64_t>& currently_logged_track_ids,
    const foxglove::messages::Timestamp& timestamp) {
  std::vector<foxglove::messages::SceneEntityDeletion> deletions;
  for (std::uint64_t previous_track_id : previously_logged_track_ids) {
    const auto found =
        std::find(currently_logged_track_ids.begin(),
                  currently_logged_track_ids.end(), previous_track_id);
    if (found != currently_logged_track_ids.end()) continue;
    foxglove::messages::SceneEntityDeletion deletion;
    deletion.timestamp = timestamp;
    deletion.type = foxglove::messages::SceneEntityDeletion::
        SceneEntityDeletionType::MATCHING_ID;
    deletion.id = std::to_string(previous_track_id);
    deletions.push_back(std::move(deletion));
  }
  return deletions;
}

// Logs one frame's transform, lidar, ego box and each track's box,
// trail, prediction and ellipse; deletes tracks absent from this one.
void write_frame(RecordingChannels& channels, const Frame& frame,
                 const std::vector<Track>& tracks,
                 const std::vector<PredictedPath>& predictions,
                 const Eigen::Vector3d& origin,
                 std::vector<std::uint64_t>& previously_logged_track_ids) {
  const std::uint64_t log_time_nanoseconds =
      to_nanoseconds(frame.capture_time_micros);
  const foxglove::messages::Timestamp timestamp =
      to_timestamp(frame.capture_time_micros);

  const Eigen::Matrix3d rotation = frame.vehicle_to_world.block<3, 3>(0, 0);
  const Eigen::Vector3d translation = frame.vehicle_to_world.block<3, 1>(0, 3);
  const Eigen::Vector3d ego_translation = translation - origin;
  const double ego_yaw = std::atan2(rotation(1, 0), rotation(0, 0));

  throw_on_error(channels.transform.log(
                     build_frame_transform(timestamp, ego_translation, ego_yaw),
                     log_time_nanoseconds),
                 "cannot log frame transform");

  throw_on_error(
      channels.points.log(build_point_cloud(frame, timestamp, origin),
                          log_time_nanoseconds),
      "cannot log point cloud");

  foxglove::messages::SceneUpdate scene_update;
  scene_update.entities.push_back(
      build_ego_entity(timestamp, ego_translation, ego_yaw));

  std::vector<std::uint64_t> currently_logged_track_ids;
  currently_logged_track_ids.reserve(tracks.size());
  for (std::size_t track_index = 0; track_index < tracks.size();
       ++track_index) {
    const Track& track = tracks[track_index];
    currently_logged_track_ids.push_back(track.track_id);
    scene_update.entities.push_back(
        build_track_entity(track, predictions[track_index], timestamp, origin));
  }

  scene_update.deletions = build_deletions(
      previously_logged_track_ids, currently_logged_track_ids, timestamp);
  previously_logged_track_ids = std::move(currently_logged_track_ids);

  throw_on_error(channels.scene.log(scene_update, log_time_nanoseconds),
                 "cannot log scene update");
}

}  // namespace

// Opens the .mcap and its three channels, predicts each track's
// path and logs all frames. Throws on any Foxglove error.
void save_replay_recording(
    const std::string& path, const SegmentLog& segment,
    const std::vector<std::vector<Track>>& confirmed_tracks_per_frame,
    const ConstantTurnRatePredictor& predictor, double horizon_seconds,
    double step_seconds) {
  foxglove::McapWriterOptions writer_options;
  writer_options.path = path;
  writer_options.compression = foxglove::McapCompression::Zstd;
  foxglove::FoxgloveResult<foxglove::McapWriter> writer_result =
      foxglove::McapWriter::create(writer_options);
  if (!writer_result.has_value())
    throw std::runtime_error(std::string("cannot create ") + path + ": " +
                             foxglove::strerror(writer_result.error()));
  foxglove::McapWriter writer = std::move(writer_result.value());

  RecordingChannels channels{
      create_channel<foxglove::messages::PointCloudChannel>("/points"),
      create_channel<foxglove::messages::FrameTransformChannel>("/tf"),
      create_channel<foxglove::messages::SceneUpdateChannel>("/scene")};

  const Eigen::Vector3d origin = recording_origin(segment);
  std::vector<std::uint64_t> previously_logged_track_ids;

  const std::size_t frame_count =
      std::min(segment.frames.size(), confirmed_tracks_per_frame.size());
  for (std::size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
    const std::vector<Track>& tracks = confirmed_tracks_per_frame[frame_index];
    std::vector<PredictedPath> predictions;
    predictions.reserve(tracks.size());
    for (const Track& track : tracks)
      predictions.push_back(
          predictor.predict(track, horizon_seconds, step_seconds));
    write_frame(channels, segment.frames[frame_index], tracks, predictions,
                origin, previously_logged_track_ids);
  }

  throw_on_error(writer.close(), "cannot close " + path);
}
