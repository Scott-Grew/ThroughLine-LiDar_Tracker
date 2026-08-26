#include "export.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include "filter.hpp"

// This file turns whatever the replay loop confirmed into a plain CSV that anything outside this
// codebase can read. It is the one place a track leaves the world frame the tracker works in and
// goes back into the vehicle frame the log itself was staged in, because that is the frame Waymo's
// own tooling expects a submission in. Nothing in the tracker, the metrics or the viewer reads
// this file back - it is written once, at the end of a run, for a Python script in the container
// to turn into Waymo's Objects proto later.

// Writes one row per confirmed track per frame, in the vehicle frame of that frame rather than the
// world frame the tracker actually reasons in, since a submission is only ever meaningful relative
// to the sensor that captured it. The two lists are walked in lockstep because confirmed track
// list at index i is exactly what the tracker confirmed for segment.frames[i].
void export_tracks(const std::string& path, const SegmentLog& segment, const std::vector<std::vector<Track>>& confirmed_tracks_per_frame) {
  std::ofstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path);
  stream << "frame_timestamp_micros,track_id,object_class,center_x,center_y,center_z,length,width,height,yaw\n";

  for (std::size_t frame_index = 0; frame_index < confirmed_tracks_per_frame.size(); ++frame_index) {
    const Frame& frame = segment.frames[frame_index];
    const Eigen::Matrix4d vehicle_from_world = frame.vehicle_to_world.inverse();
    const double heading_offset = std::atan2(frame.vehicle_to_world(1, 0), frame.vehicle_to_world(0, 0));

    for (const Track& track : confirmed_tracks_per_frame[frame_index]) {
      const Eigen::Vector4d world_position(track.state.mean(0), track.state.mean(1), track.state.center_z, 1.0);
      const Eigen::Vector4d vehicle_position = vehicle_from_world * world_position;
      const double vehicle_yaw = wrap_angle(track.state.mean(2) - heading_offset);

      stream << frame.capture_time_micros << ','
             << track.track_id << ','
             << static_cast<int>(track.object_class) << ','
             << vehicle_position(0) << ','
             << vehicle_position(1) << ','
             << vehicle_position(2) << ','
             << track.state.length << ','
             << track.state.width << ','
             << track.state.height << ','
             << vehicle_yaw << '\n';
    }
  }
}
