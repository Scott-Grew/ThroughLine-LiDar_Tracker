// Publishes a run's lidar points, track boxes, trails, forecasts and
// uncertainty ellipses as Foxglove scene messages, live and to MCAP.

#include "scene.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <span>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <foxglove/error.hpp>
#include <foxglove/mcap.hpp>
#include <foxglove/messages.hpp>
#include <foxglove/websocket.hpp>
#include "filter.hpp"
#include "types.hpp"

namespace fg = foxglove::messages;
using LineType = fg::LinePrimitive::LineType;
using Points = std::vector<fg::Point3>;

namespace {

// How far ahead the scene draws each track's forecast, and the spacing of
// its points. Chosen.
constexpr std::int64_t kForecastHorizonMicros = 3'000'000;
constexpr std::int64_t kForecastStepMicros = 100'000;

// Drawing constants for the point height ramp, the ellipse, and the line,
// label and ego box sizes. Lengths are in metres. Chosen.
constexpr float kPointHeightLow = -2.0F;
constexpr float kPointHeightHigh = 4.0F;
constexpr int kEllipseSegments = 32;
constexpr double kEllipseSigma = 2.0;
constexpr double kLineThickness = 0.08;
constexpr double kLabelFontSize = 0.6;
constexpr fg::Vector3 kEgoSize{.x = 5.2, .y = 2.4, .z = 1.8};
constexpr fg::Color kEgoColor{.r = 0.8, .g = 0.8, .b = 0.8, .a = 1.0};
constexpr fg::Color kWhite{.r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0};
constexpr double kBoxAlpha = 0.35;

// Bytes per packed point, x, y and z as float32 then rgba as uint32.
constexpr std::uint32_t kPointStrideBytes = 16;
constexpr std::uint32_t kColorOffsetBytes = 12;
constexpr std::int64_t kNanosPerMicro = 1'000;

fg::Color class_color(ObjectClass object_class) {
  switch (object_class) {
    case ObjectClass::Vehicle:
      return {.r = 0.0, .g = 0.59, .b = 1.0, .a = 1.0};
    case ObjectClass::Pedestrian:
      return {.r = 1.0, .g = 0.55, .b = 0.0, .a = 1.0};
    case ObjectClass::Cyclist:
      return {.r = 0.0, .g = 0.86, .b = 0.47, .a = 1.0};
  }
  return kWhite;
}

fg::Timestamp to_timestamp(std::int64_t capture_time_micros) {
  const std::int64_t seconds = capture_time_micros / kMicrosPerSecond;
  const std::int64_t nanos =
      (capture_time_micros % kMicrosPerSecond) * kNanosPerMicro;
  return {.sec = static_cast<std::uint32_t>(seconds),
          .nsec = static_cast<std::uint32_t>(nanos)};
}

std::uint64_t to_nanoseconds(std::int64_t capture_time_micros) {
  return static_cast<std::uint64_t>(capture_time_micros * kNanosPerMicro);
}

fg::Vector3 to_vector(const Eigen::Vector3d& vector) {
  return {.x = vector.x(), .y = vector.y(), .z = vector.z()};
}

fg::Point3 to_point(const Eigen::Vector2d& position, double height) {
  return {.x = position.x(), .y = position.y(), .z = height};
}

// A rotation of yaw about z is the quaternion (0, 0, sin(yaw/2), cos(yaw/2)).
fg::Quaternion yaw_rotation(double yaw) {
  return {
      .x = 0.0, .y = 0.0, .z = std::sin(yaw / 2.0), .w = std::cos(yaw / 2.0)};
}

fg::Pose pose_at(const Eigen::Vector3d& position, double yaw) {
  fg::Pose pose;
  pose.position = to_vector(position);
  pose.orientation = yaw_rotation(yaw);
  return pose;
}

// Colour of a point by height, as 0xaarrggbb, the order Lichtblick reads.
// The ramp runs blue to green over the lower half of the height range and
// green to yellow over the upper half, sampled into a table at compile time.
constexpr std::size_t kRampSteps = 256;

constexpr std::uint32_t ramp_color(float fraction) {
  const bool lower_half = fraction < 0.5F;
  const float local = lower_half ? fraction * 2.0F : (fraction - 0.5F) * 2.0F;
  const float red = lower_half ? 0.0F : local;
  const float green = lower_half ? local : 1.0F;
  const float blue = lower_half ? 1.0F - local : 0.0F;
  const auto red_byte = static_cast<std::uint32_t>(red * 255.0F);
  const auto green_byte = static_cast<std::uint32_t>(green * 255.0F);
  const auto blue_byte = static_cast<std::uint32_t>(blue * 255.0F);
  return (0xFFU << 24U) | (red_byte << 16U) | (green_byte << 8U) | blue_byte;
}

constexpr std::array<std::uint32_t, kRampSteps> make_height_ramp() {
  std::array<std::uint32_t, kRampSteps> table{};
  for (std::size_t step = 0; step < kRampSteps; ++step)
    table.at(step) = ramp_color(static_cast<float>(step) / (kRampSteps - 1));
  return table;
}

constexpr std::array<std::uint32_t, kRampSteps> kHeightRamp =
    make_height_ramp();

std::uint32_t height_color(float height) {
  const float clamped = std::clamp(height, kPointHeightLow, kPointHeightHigh);
  const float fraction =
      (clamped - kPointHeightLow) / (kPointHeightHigh - kPointHeightLow);
  return kHeightRamp.at(static_cast<std::size_t>(fraction * (kRampSteps - 1)));
}

// The packed point layout, built once.
const std::vector<fg::PackedElementField>& point_fields() {
  using Field = fg::PackedElementField;
  static const std::vector<Field> fields = {
      {.name = "x", .offset = 0, .type = Field::NumericType::FLOAT32},
      {.name = "y", .offset = 4, .type = Field::NumericType::FLOAT32},
      {.name = "z", .offset = 8, .type = Field::NumericType::FLOAT32},
      {.name = "rgba",
       .offset = kColorOffsetBytes,
       .type = Field::NumericType::UINT32},
  };
  return fields;
}

fg::LinePrimitive line(LineType type, fg::Color color, Points points) {
  fg::LinePrimitive primitive;
  primitive.type = type;
  primitive.thickness = kLineThickness;
  primitive.color = color;
  primitive.points = std::move(points);
  return primitive;
}

// The 2-sigma position ellipse of a forecast state, drawn at its height.
// Each ring point is centre + V (k sqrt(lambda) u) for eigenvectors V and
// values lambda of the covariance, k = kEllipseSigma and unit-circle u.
Points ellipse(const TrackState& end, const Eigen::Vector3d& origin) {
  const Eigen::Vector2d center = end.position() - origin.head<2>();
  const Eigen::Matrix2d covariance = end.covariance.topLeftCorner<2, 2>();
  const double height = end.center_z - origin.z();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
  const Eigen::Vector2d axes =
      kEllipseSigma * solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  Points points;
  points.reserve(kEllipseSegments);
  for (int segment = 0; segment < kEllipseSegments; ++segment) {
    const double angle = 2.0 * std::numbers::pi * segment / kEllipseSegments;
    const Eigen::Vector2d unit(std::cos(angle), std::sin(angle));
    const Eigen::Vector2d offset =
        solver.eigenvectors() * axes.cwiseProduct(unit);
    points.push_back(to_point(center + offset, height));
  }
  return points;
}

void check(foxglove::FoxgloveError error, const std::string& action) {
  if (error != foxglove::FoxgloveError::Ok)
    throw std::runtime_error(action + ": " + foxglove::strerror(error));
}

template <typename Value>
Value unwrap(foxglove::FoxgloveResult<Value> result, const std::string& what) {
  if (!result.has_value())
    throw std::runtime_error(what + ": " + foxglove::strerror(result.error()));
  return std::move(result.value());
}

template <typename Channel>
Channel open_channel(const std::string& topic) {
  return unwrap(Channel::create(topic), "cannot create " + topic);
}

}  // namespace

// The sinks every channel logs to, plus the channels. Both sinks share the
// default context, so one log() call reaches the viewer and the file.
struct Scene::Sinks {
  std::optional<foxglove::WebSocketServer> server;
  std::optional<foxglove::McapWriter> writer;
  fg::PointCloudChannel points = open_channel<fg::PointCloudChannel>("/points");
  fg::FrameTransformChannel transform =
      open_channel<fg::FrameTransformChannel>("/tf");
  fg::SceneUpdateChannel scene = open_channel<fg::SceneUpdateChannel>("/scene");
};

Scene::Scene(const SceneOptions& options, FilterNoise noise)
    : filter_(noise), sinks_(std::make_unique<Sinks>()) {
  if (options.live_port != 0) {
    foxglove::WebSocketServerOptions server_options;
    server_options.name = "tracker";
    server_options.port = options.live_port;
    sinks_->server =
        unwrap(foxglove::WebSocketServer::create(std::move(server_options)),
               "cannot start the live server");
  }
  if (!options.recording_path.empty()) {
    foxglove::McapWriterOptions writer_options;
    writer_options.path = options.recording_path;
    writer_options.compression = foxglove::McapCompression::Zstd;
    sinks_->writer = unwrap(foxglove::McapWriter::create(writer_options),
                            "cannot create " + options.recording_path);
  }
}

// A publisher destroyed without close() still finishes its file, but the
// destructor cannot report a failure, so main.cpp calls close() itself.
Scene::~Scene() {
  if (!sinks_) return;
  if (sinks_->writer) sinks_->writer->close();
  if (sinks_->server) sinks_->server->stop();
}

void Scene::close() {
  if (sinks_->writer) check(sinks_->writer->close(), "cannot close recording");
  if (sinks_->server) check(sinks_->server->stop(), "cannot stop live server");
  sinks_.reset();
}

void Scene::publish(const Frame& frame, const std::vector<Track>& tracks) {
  const Eigen::Matrix3d rotation = frame.vehicle_to_world.topLeftCorner<3, 3>();
  const Eigen::Vector3d translation =
      frame.vehicle_to_world.topRightCorner<3, 1>();
  if (!has_origin_) {
    origin_ = translation;
    has_origin_ = true;
  }
  const fg::Timestamp stamp = to_timestamp(frame.capture_time_micros);
  const std::uint64_t log_time = to_nanoseconds(frame.capture_time_micros);
  const Eigen::Vector3d ego_position = translation - origin_;
  const Eigen::Vector2d origin_xy = origin_.head<2>();
  const double ego_yaw = vehicle_yaw(frame.vehicle_to_world);

  fg::FrameTransform transform;
  transform.timestamp = stamp;
  transform.parent_frame_id = "world";
  transform.child_frame_id = "ego";
  transform.translation = to_vector(ego_position);
  transform.rotation = yaw_rotation(ego_yaw);
  check(sinks_->transform.log(transform, log_time), "cannot log transform");

  // Each packed point is x, y, z then its height colour. The transform
  // writes straight into the packed buffer through a strided map.
  using PointMatrix = Eigen::Matrix<float, 3, Eigen::Dynamic>;
  using PackedPoints =
      Eigen::Map<PointMatrix, Eigen::Unaligned, Eigen::OuterStride<4>>;
  const auto point_count = static_cast<Eigen::Index>(frame.points.size());
  const Eigen::Map<const PointMatrix> local_points(&frame.points.data()->x, 3,
                                                   point_count);
  const Eigen::Vector3f shift = ego_position.cast<float>();
  fg::PointCloud cloud;
  cloud.timestamp = stamp;
  cloud.frame_id = "world";
  cloud.pose = pose_at(Eigen::Vector3d::Zero(), 0.0);
  cloud.point_stride = kPointStrideBytes;
  cloud.fields = point_fields();
  packed_points_.resize(static_cast<std::size_t>(point_count) * 4);
  PackedPoints scene_points(packed_points_.data(), 3, point_count);
  scene_points = (rotation.cast<float>() * local_points).colwise() + shift;
  for (Eigen::Index index = 0; index < point_count; ++index) {
    const std::uint32_t color = height_color(scene_points(2, index));
    packed_points_[index * 4 + 3] = std::bit_cast<float>(color);
  }
  const std::span<const std::byte> bytes =
      std::as_bytes(std::span(packed_points_));
  cloud.data.assign(bytes.begin(), bytes.end());
  check(sinks_->points.log(cloud, log_time), "cannot log point cloud");

  fg::SceneUpdate update;
  fg::SceneEntity& ego = update.entities.emplace_back();
  ego.timestamp = stamp;
  ego.frame_id = "world";
  ego.id = "ego";
  fg::CubePrimitive& ego_cube = ego.cubes.emplace_back();
  ego_cube.pose = pose_at(ego_position, ego_yaw);
  ego_cube.size = kEgoSize;
  ego_cube.color = kEgoColor;

  std::vector<std::uint64_t> current_track_ids;
  current_track_ids.reserve(tracks.size());
  for (const Track& track : tracks) {
    current_track_ids.push_back(track.track_id);
    const fg::Color color = class_color(track.object_class);
    const Eigen::Vector2d position = track.state.position() - origin_xy;
    const double height = track.state.center_z - origin_.z();
    const Eigen::Vector3d center(position.x(), position.y(), height);

    fg::SceneEntity& entity = update.entities.emplace_back();
    entity.timestamp = stamp;
    entity.frame_id = "world";
    entity.id = std::to_string(track.track_id);

    fg::CubePrimitive& cube = entity.cubes.emplace_back();
    cube.pose = pose_at(center, track.state.yaw());
    cube.size = {.x = track.state.length,
                 .y = track.state.width,
                 .z = track.state.height};
    cube.color = {.r = color.r, .g = color.g, .b = color.b, .a = kBoxAlpha};

    Points trail;
    trail.reserve(track.history.size());
    for (const Eigen::Vector2d& past : track.history)
      trail.push_back(to_point(past - origin_xy, height));
    entity.lines.push_back(line(LineType::LINE_STRIP, color, std::move(trail)));

    TrackState forecast = track.state;
    const std::int64_t horizon = forecast.time_micros + kForecastHorizonMicros;
    Points path;
    while (forecast.time_micros < horizon) {
      filter_.advance(forecast, forecast.time_micros + kForecastStepMicros);
      path.push_back(to_point(forecast.position() - origin_xy, height));
    }
    entity.lines.push_back(line(LineType::LINE_STRIP, color, std::move(path)));
    entity.lines.push_back(
        line(LineType::LINE_LOOP, color, ellipse(forecast, origin_)));

    fg::TextPrimitive& label = entity.texts.emplace_back();
    const Eigen::Vector3d roof(0.0, 0.0, track.state.height / 2.0);
    label.pose = pose_at(center + roof, 0.0);
    label.billboard = true;
    label.font_size = kLabelFontSize;
    label.scale_invariant = true;
    label.color = color;
    label.text = std::to_string(track.track_id);
  }

  // A track drawn last frame and absent now is deleted, or the viewer keeps
  // drawing it.
  for (const std::uint64_t previous_id : drawn_track_ids_) {
    const bool still_drawn =
        std::ranges::find(current_track_ids, previous_id) !=
        current_track_ids.end();
    if (still_drawn) continue;
    fg::SceneEntityDeletion& deletion = update.deletions.emplace_back();
    deletion.timestamp = stamp;
    deletion.type =
        fg::SceneEntityDeletion::SceneEntityDeletionType::MATCHING_ID;
    deletion.id = std::to_string(previous_id);
  }
  drawn_track_ids_ = std::move(current_track_ids);
  check(sinks_->scene.log(update, log_time), "cannot log scene");
}
