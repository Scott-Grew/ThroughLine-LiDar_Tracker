#include "iou.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

// This file answers how much two rotated 3d boxes overlap, as a single number between 0 and 1.
// The metrics code calls it to decide whether a track's box is close enough to a ground truth
// box to count as the same object, using the thresholds Waymo itself uses for scoring. A box is
// treated as a flat rotated rectangle seen from above, plus a height range; the two rectangles are
// clipped against each other to get the overlapping footprint, that footprint is combined with
// however much the two boxes overlap vertically, and the resulting overlap volume is compared
// against the volume the two boxes cover between them.

namespace {

constexpr double kEpsilon = 1e-9;

// The four ground corners of a box seen from above, centred on the box and turned by its heading,
// listed counter-clockwise starting at the front right corner. Length runs along the way the box
// is facing, width runs across it. Both the clipping code and the overlap test below only ever
// look at boxes through this flattened, rotated rectangle.
std::array<Eigen::Vector2d, 4> footprint(const Box& box) {
  const double half_length = box.length / 2.0;
  const double half_width = box.width / 2.0;
  const std::array<Eigen::Vector2d, 4> local_corners = {
      Eigen::Vector2d(half_length, -half_width),
      Eigen::Vector2d(half_length, half_width),
      Eigen::Vector2d(-half_length, half_width),
      Eigen::Vector2d(-half_length, -half_width),
  };
  const double cosine = std::cos(box.yaw);
  const double sine = std::sin(box.yaw);
  const Eigen::Vector2d centre(box.center_x, box.center_y);
  std::array<Eigen::Vector2d, 4> world_corners;
  for (std::size_t index = 0; index < local_corners.size(); ++index) {
    const Eigen::Vector2d& local = local_corners[index];
    world_corners[index] = centre + Eigen::Vector2d(cosine * local.x() - sine * local.y(), sine * local.x() + cosine * local.y());
  }
  return world_corners;
}

// How far to one side of a line a point sits, using the sign to tell one side of the line from the
// other. Both the clipping test below and the corner ordering above rely on this same convention.
double side_of_line(const Eigen::Vector2d& edge_start, const Eigen::Vector2d& edge_end, const Eigen::Vector2d& point) {
  const Eigen::Vector2d edge_vector = edge_end - edge_start;
  const Eigen::Vector2d point_vector = point - edge_start;
  return edge_vector.x() * point_vector.y() - edge_vector.y() * point_vector.x();
}

// One step of Sutherland-Hodgman polygon clipping: cuts the given polygon down to whichever part
// lies on the same side of one edge as the reference point does, inserting a new vertex wherever
// the polygon boundary crosses that edge. Called once per edge of the other box's footprint, so
// four calls in a row clip a whole rectangle down to its overlap with another rectangle. A
// polygon that has already been clipped away to nothing is passed through unchanged.
std::vector<Eigen::Vector2d> clip_polygon(std::vector<Eigen::Vector2d> polygon, Eigen::Vector2d edge_start, Eigen::Vector2d edge_end, Eigen::Vector2d inside_reference) {
  if (polygon.empty()) return polygon;
  const double reference_side = side_of_line(edge_start, edge_end, inside_reference);
  std::vector<Eigen::Vector2d> clipped;
  for (std::size_t index = 0; index < polygon.size(); ++index) {
    const Eigen::Vector2d& current = polygon[index];
    const Eigen::Vector2d& next = polygon[(index + 1) % polygon.size()];
    const double current_side = side_of_line(edge_start, edge_end, current);
    const double next_side = side_of_line(edge_start, edge_end, next);
    const bool current_inside = reference_side >= 0.0 ? current_side >= -kEpsilon : current_side <= kEpsilon;
    const bool next_inside = reference_side >= 0.0 ? next_side >= -kEpsilon : next_side <= kEpsilon;
    if (current_inside) clipped.push_back(current);
    if (current_inside != next_inside && std::abs(current_side - next_side) > kEpsilon) {
      const double crossing_fraction = current_side / (current_side - next_side);
      clipped.push_back(current + crossing_fraction * (next - current));
    }
  }
  return clipped;
}

// The area enclosed by a polygon, by the shoelace formula. A polygon that clipping has reduced to
// fewer than three points has no area left to enclose.
double polygon_area(const std::vector<Eigen::Vector2d>& polygon) {
  if (polygon.size() < 3) return 0.0;
  double signed_area = 0.0;
  for (std::size_t index = 0; index < polygon.size(); ++index) {
    const Eigen::Vector2d& current = polygon[index];
    const Eigen::Vector2d& next = polygon[(index + 1) % polygon.size()];
    signed_area += current.x() * next.y() - next.x() * current.y();
  }
  return std::abs(signed_area) * 0.5;
}

}

// The one function the rest of the codebase calls. Clips the first box's footprint against every
// edge of the second box's footprint to get the overlapping ground area, multiplies that by
// however much the two boxes overlap in height, and divides by the combined volume of both boxes
// minus that overlap. Two boxes that do not touch at all, on the ground or in height, score zero;
// identical boxes score one.
double intersection_over_union_3d(const Box& first, const Box& second) {
  const std::array<Eigen::Vector2d, 4> first_footprint = footprint(first);
  const std::array<Eigen::Vector2d, 4> second_footprint = footprint(second);
  const Eigen::Vector2d second_centre(second.center_x, second.center_y);
  std::vector<Eigen::Vector2d> overlap(first_footprint.begin(), first_footprint.end());
  for (std::size_t index = 0; index < second_footprint.size(); ++index) {
    overlap = clip_polygon(overlap, second_footprint[index], second_footprint[(index + 1) % second_footprint.size()], second_centre);
  }

  const double first_bottom = first.center_z - first.height / 2.0;
  const double first_top = first.center_z + first.height / 2.0;
  const double second_bottom = second.center_z - second.height / 2.0;
  const double second_top = second.center_z + second.height / 2.0;
  const double vertical_overlap = std::max(0.0, std::min(first_top, second_top) - std::max(first_bottom, second_bottom));

  const double intersection_volume = polygon_area(overlap) * vertical_overlap;
  const double first_volume = first.length * first.width * first.height;
  const double second_volume = second.length * second.width * second.height;
  const double union_volume = first_volume + second_volume - intersection_volume;

  return union_volume < 1e-9 ? 0.0 : intersection_volume / union_volume;
}
