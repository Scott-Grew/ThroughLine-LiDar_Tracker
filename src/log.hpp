#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "types.hpp"

struct SegmentLog {
  std::string segment_name;
  std::vector<Frame> frames;
};

SegmentLog read_segment_log(const std::string& path);
void write_segment_log(const std::string& path, const SegmentLog& segment);
