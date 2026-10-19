// Declares the publisher that streams each frame's lidar points and
// confirmed tracks to a live Lichtblick viewer and, when asked, to an .mcap.

#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "filter.hpp"
#include "types.hpp"

// Where the scene goes. A port of zero means no live server, and an empty
// path means no recording.
struct SceneOptions {
  std::uint16_t live_port = 0;
  std::string recording_path;
};

// Owns the live server and the recording. publish() sends one frame to both;
// close() finishes the recording and stops the server.
class Scene {
 public:
  Scene(const SceneOptions& options, FilterNoise noise);
  ~Scene();
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;
  Scene(Scene&&) = default;
  Scene& operator=(Scene&&) = default;
  void publish(const Frame& frame, const std::vector<Track>& tracks);
  void close();

 private:
  struct Sinks;
  MotionFilter filter_;
  std::unique_ptr<Sinks> sinks_;
  // Frame 0's position, subtracted from every published coordinate so the
  // scene stays within float precision.
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  bool has_origin_ = false;
  std::vector<std::uint64_t> drawn_track_ids_;
};
