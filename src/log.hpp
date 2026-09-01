#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "types.hpp"

struct SegmentLog {
  std::string segment_name;
  std::vector<Frame> frames;
  double measured_position_sigma = 0.0;
  double measured_yaw_sigma = 0.0;
};

SegmentLog read_segment_log(const std::string& path);
void write_segment_log(const std::string& path,
                       const SegmentLog& segment);
