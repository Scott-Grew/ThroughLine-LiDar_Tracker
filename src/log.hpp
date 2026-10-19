// Declares the reader of a staged segment, an MCAP file holding one Segment
// message and one Frame per lidar frame, both from proto/tracker.proto.

#pragma once
#include <memory>
#include <optional>
#include <string>
#include "types.hpp"

// The segment's name and the detector's measured position and yaw error,
// which seed the filter's measurement noise.
struct SegmentInfo {
  std::string name;
  double position_sigma = 0.0;
  double yaw_sigma = 0.0;
};

// Opens a staged segment and hands out its frames one at a time, in time
// order, so the whole log never sits in memory.
class SegmentReader {
 public:
  explicit SegmentReader(const std::string& path);
  ~SegmentReader();
  SegmentReader(const SegmentReader&) = delete;
  SegmentReader& operator=(const SegmentReader&) = delete;
  SegmentReader(SegmentReader&&) = default;
  SegmentReader& operator=(SegmentReader&&) = default;
  const SegmentInfo& info() const { return info_; }
  std::optional<Frame> next_frame();

 private:
  struct Cursor;
  SegmentInfo info_;
  std::unique_ptr<Cursor> cursor_;
};
