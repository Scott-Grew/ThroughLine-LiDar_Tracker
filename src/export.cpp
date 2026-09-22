// Writes confirmed tracks to a CSV in each frame's vehicle frame;
// the scripts in eval/ read it to score the run.

#include "export.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include "filter.hpp"

// Writes one row per confirmed track per frame, converting each
// track from the world frame back into that frame's vehicle frame.
void export_tracks(
    const std::string& path, const SegmentLog& segment,
    const std::vector<std::vector<Track>>& confirmed_tracks_per_frame) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path);
  stream << "frame_timestamp_micros,track_id,object_class,center_x,"
            "center_y,center_z,length,width,height,yaw\n";

  for (std::size_t frame_index = 0;
       frame_index < confirmed_tracks_per_frame.size(); ++frame_index) {
    const Frame& frame = segment.frames[frame_index];
    const Eigen::Matrix4d vehicle_from_world = frame.vehicle_to_world.inverse();
    const double ego_yaw =
        std::atan2(frame.vehicle_to_world(1, 0), frame.vehicle_to_world(0, 0));

    for (const Track& track : confirmed_tracks_per_frame[frame_index]) {
      const Eigen::Vector4d world_position(track.state.mean(kPositionXIndex),
                                           track.state.mean(kPositionYIndex),
                                           track.state.center_z, 1.0);
      const Eigen::Vector4d vehicle_position =
          vehicle_from_world * world_position;
      // Rotates the track's world yaw into this frame's vehicle frame yaw.
      const double yaw_in_vehicle_frame =
          wrap_angle(track.state.mean(kYawIndex) - ego_yaw);

      stream << frame.capture_time_micros << ',' << track.track_id << ','
             << static_cast<int>(track.object_class) << ','
             << vehicle_position(0) << ',' << vehicle_position(1) << ','
             << vehicle_position(2) << ',' << track.state.length << ','
             << track.state.width << ',' << track.state.height << ','
             << yaw_in_vehicle_frame << '\n';
    }
  }
}
