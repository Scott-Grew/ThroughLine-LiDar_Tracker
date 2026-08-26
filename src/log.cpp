#include "log.hpp"

SegmentLog read_segment_log(const std::string& path) {
  (void)path;
  return SegmentLog{};
}

void write_segment_log(const std::string& path, const SegmentLog& segment) {
  (void)path;
  (void)segment;
}
