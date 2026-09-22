// Declares the in-memory shape of a staged segment and the
// reader/writer for its binary file, defined in log.cpp.

#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "types.hpp"

// One staged segment: its frames plus the box-jitter sigmas the
// stager measured for it.
struct SegmentLog {
  std::string segment_name;
  std::vector<Frame> frames;
  double measured_position_sigma = 0.0;
  double measured_yaw_sigma = 0.0;
};

// Reads the segment staged at path; throws if it isn't a current-layout log.
SegmentLog read_segment_log(const std::string& path);
// Writes segment to path in the staged binary layout, host byte order.
void write_segment_log(const std::string& path, const SegmentLog& segment);
