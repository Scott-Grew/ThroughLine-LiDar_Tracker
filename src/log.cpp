#include "log.hpp"
#include <fstream>
#include <stdexcept>

// This file is the only place the staged binary log format is read or written. stage_segment.py
// writes it from Waymo's parquet components, and write_segment_log/read_segment_log here are its
// only reader and writer on the C++ side - everything downstream works with a SegmentLog in
// memory and never touches the bytes directly. The layout is little-endian throughout: a fixed
// header, then one record per frame holding that frame's pose, its lidar points as plain x, y, z
// float32 triples, and its ground truth boxes. The magic string is bumped whenever the layout
// changes, so a log staged under an older layout is refused outright instead of being silently
// misread.

namespace {

constexpr char kMagic[8] = {'T', 'R', 'K', 'L', 'O', 'G', '0', '2'};

// Writes one value of any fixed-size type as its raw bytes. Every scalar field in the format goes
// through this, so the layout below is exactly the memory layout of whatever type is passed in.
template <typename Value>
void write_value(std::ofstream& stream, const Value& value) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(Value));
}

// The other half of write_value: reads one value's raw bytes back out. Throws if the stream ran
// out before a full value was available, which is what turns a truncated file into a clear error
// instead of a silently half-read struct.
template <typename Value>
Value read_value(std::ifstream& stream) {
  Value value;
  stream.read(reinterpret_cast<char*>(&value), sizeof(Value));
  if (!stream) throw std::runtime_error("segment log truncated");
  return value;
}

// Writes the seven numbers a Box carries, in the fixed order read_box below expects back.
void write_box(std::ofstream& stream, const Box& box) {
  write_value(stream, box.center_x);
  write_value(stream, box.center_y);
  write_value(stream, box.center_z);
  write_value(stream, box.length);
  write_value(stream, box.width);
  write_value(stream, box.height);
  write_value(stream, box.yaw);
}

// Reads a Box back in the same order write_box wrote it.
Box read_box(std::ifstream& stream) {
  Box box;
  box.center_x = read_value<double>(stream);
  box.center_y = read_value<double>(stream);
  box.center_z = read_value<double>(stream);
  box.length = read_value<double>(stream);
  box.width = read_value<double>(stream);
  box.height = read_value<double>(stream);
  box.yaw = read_value<double>(stream);
  return box;
}

}

// Writes a whole segment: the magic and header first, then one record per frame - its capture
// time, its 4x4 pose in row-major order, its points as raw x, y, z float32 triples, and its
// ground truth boxes. stage_segment.py writes the exact same layout from Python, and the two are
// checked against each other by round-tripping a staged segment through both.
void write_segment_log(const std::string& path, const SegmentLog& segment) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  stream.write(kMagic, sizeof(kMagic));
  write_value(stream, static_cast<std::uint32_t>(segment.frames.size()));
  write_value(stream, static_cast<std::uint32_t>(segment.segment_name.size()));
  stream.write(segment.segment_name.data(), segment.segment_name.size());
  for (const Frame& frame : segment.frames) {
    write_value(stream, frame.capture_time_micros);
    for (int row = 0; row < 4; ++row)
      for (int column = 0; column < 4; ++column) write_value(stream, frame.vehicle_to_world(row, column));
    write_value(stream, static_cast<std::uint32_t>(frame.points.size()));
    stream.write(reinterpret_cast<const char*>(frame.points.data()), frame.points.size() * sizeof(Point));
    write_value(stream, static_cast<std::uint32_t>(frame.ground_truth.size()));
    for (const GroundTruthBox& truth : frame.ground_truth) {
      write_value(stream, truth.object_id);
      write_value(stream, static_cast<std::uint8_t>(truth.object_class));
      write_box(stream, truth.box);
      write_value(stream, truth.lidar_points_in_box);
    }
  }
}

// Reads a whole segment back, refusing anything whose magic does not match this file's current
// layout - which is what makes a log staged under an older format fail loudly here rather than
// being read as if its bytes meant something they do not.
SegmentLog read_segment_log(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  char magic[8];
  stream.read(magic, sizeof(magic));
  if (!stream || std::string(magic, 8) != std::string(kMagic, 8)) throw std::runtime_error("not a segment log: " + path);
  SegmentLog segment;
  const auto frame_count = read_value<std::uint32_t>(stream);
  const auto name_length = read_value<std::uint32_t>(stream);
  segment.segment_name.resize(name_length);
  stream.read(segment.segment_name.data(), name_length);
  segment.frames.resize(frame_count);
  for (Frame& frame : segment.frames) {
    frame.capture_time_micros = read_value<std::int64_t>(stream);
    for (int row = 0; row < 4; ++row)
      for (int column = 0; column < 4; ++column) frame.vehicle_to_world(row, column) = read_value<double>(stream);
    frame.points.resize(read_value<std::uint32_t>(stream));
    stream.read(reinterpret_cast<char*>(frame.points.data()), frame.points.size() * sizeof(Point));
    frame.ground_truth.resize(read_value<std::uint32_t>(stream));
    for (GroundTruthBox& truth : frame.ground_truth) {
      truth.object_id = read_value<std::uint64_t>(stream);
      truth.object_class = static_cast<ObjectClass>(read_value<std::uint8_t>(stream));
      truth.box = read_box(stream);
      truth.lidar_points_in_box = read_value<std::int32_t>(stream);
    }
    if (!stream) throw std::runtime_error("segment log truncated");
  }
  return segment;
}
