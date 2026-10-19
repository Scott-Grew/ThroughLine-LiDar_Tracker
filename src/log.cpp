// Reads a staged segment with the MCAP library and protobuf; main.cpp pulls
// one frame at a time from it.

#define MCAP_IMPLEMENTATION
#include "log.hpp"

#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <mcap/errors.hpp>
#include <mcap/reader.hpp>
#include <mcap/types.hpp>
#include "tracker.pb.h"
#include "types.hpp"

namespace {

Box to_box(const tracker::Box& message) {
  Box box;
  box.center_x = message.center_x();
  box.center_y = message.center_y();
  box.center_z = message.center_z();
  box.length = message.length();
  box.width = message.width();
  box.height = message.height();
  box.yaw = message.yaw();
  return box;
}

Frame to_frame(const tracker::Frame& message) {
  using RowMajorPose = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
  Frame frame;
  frame.capture_time_micros = message.capture_time_micros();
  frame.vehicle_to_world =
      Eigen::Map<const RowMajorPose>(message.vehicle_to_world().data());
  // The log stores x, y, z triples, the same layout as Point.
  static_assert(sizeof(Point) == 3 * sizeof(float));
  frame.points.resize(message.points_size() / 3);
  std::memcpy(frame.points.data(), message.points().data(),
              frame.points.size() * sizeof(Point));
  frame.detections.reserve(message.detections_size());
  for (const tracker::Detection& detection : message.detections()) {
    const auto object_class =
        static_cast<ObjectClass>(detection.object_class());
    frame.detections.push_back(
        {.object_class = object_class, .box = to_box(detection.box())});
  }
  return frame;
}

template <typename Message>
Message parse(const mcap::Message& record, const std::string& path) {
  Message message;
  const auto size = static_cast<int>(record.dataSize);
  if (!message.ParseFromArray(record.data, size))
    throw std::runtime_error(path + ": cannot parse a message");
  return message;
}

// The message view and the iterator into it, built in this order because
// the iterators refer back to the view.
struct Messages {
  explicit Messages(mcap::LinearMessageView all_messages)
      : view(std::move(all_messages)), next(view.begin()), end(view.end()) {}
  mcap::LinearMessageView view;
  mcap::LinearMessageView::Iterator next;
  mcap::LinearMessageView::Iterator end;
};

}  // namespace

// The open file, the first problem the library reported while reading it,
// and the messages; all live as long as the SegmentReader.
struct SegmentReader::Cursor {
  std::string path;
  std::string problem;
  mcap::McapReader reader;
  std::unique_ptr<Messages> messages;

  void throw_on_problem() const {
    if (!problem.empty()) throw std::runtime_error(path + ": " + problem);
  }
};

SegmentReader::SegmentReader(const std::string& path)
    : cursor_(std::make_unique<Cursor>()) {
  cursor_->path = path;
  const mcap::Status opened = cursor_->reader.open(path);
  if (!opened.ok()) throw std::runtime_error(path + ": " + opened.message);
  const auto remember = [cursor = cursor_.get()](const mcap::Status& status) {
    if (cursor->problem.empty()) cursor->problem = status.message;
  };
  cursor_->messages =
      std::make_unique<Messages>(cursor_->reader.readMessages(remember));
  cursor_->throw_on_problem();

  Messages& messages = *cursor_->messages;
  const bool has_header = messages.next != messages.end &&
                          messages.next->channel->topic == "/segment";
  if (!has_header)
    throw std::runtime_error(path + ": no /segment message at the start");
  const auto header = parse<tracker::Segment>(messages.next->message, path);
  info_ = {.name = header.name(),
           .position_sigma = header.position_sigma(),
           .yaw_sigma = header.yaw_sigma()};
  ++messages.next;
  cursor_->throw_on_problem();
}

SegmentReader::~SegmentReader() = default;

std::optional<Frame> SegmentReader::next_frame() {
  Messages& messages = *cursor_->messages;
  if (messages.next == messages.end) return std::nullopt;
  const auto message =
      parse<tracker::Frame>(messages.next->message, cursor_->path);
  ++messages.next;
  cursor_->throw_on_problem();
  return to_frame(message);
}
