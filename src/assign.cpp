#include "assign.hpp"

#include <algorithm>
#include <vector>
#include <dlib/optimization/max_cost_assignment.h>

// Matches a track/detection cost grid into pairs, by a greedy
// pass or dlib's Hungarian solver; the tracker picks which.

namespace {

// One candidate pairing and its cost, used only while the greedy
// method walks options from cheapest to most expensive.
struct Candidate {
  double cost;
  int row;
  int column;
};

// Takes pairings under the gate cheapest first, keeping ones whose
// row and column are still free; can miss a cheaper overall match.
std::vector<std::pair<int, int>> greedy_pairs(
    const Eigen::MatrixXd& cost, double gate) {
  std::vector<Candidate> candidates;
  for (int row = 0; row < cost.rows(); ++row)
    for (int column = 0; column < cost.cols(); ++column)
      if (cost(row, column) <= gate)
        candidates.push_back({cost(row, column), row, column});
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& first, const Candidate& second) {
              if (first.cost != second.cost)
                return first.cost < second.cost;
              if (first.row != second.row)
                return first.row < second.row;
              return first.column < second.column;
            });
  std::vector<bool> row_used(cost.rows(), false),
      column_used(cost.cols(), false);
  std::vector<std::pair<int, int>> pairs;
  for (const Candidate& candidate : candidates) {
    if (row_used[candidate.row] || column_used[candidate.column])
      continue;
    row_used[candidate.row] = true;
    column_used[candidate.column] = true;
    pairs.emplace_back(candidate.row, candidate.column);
  }
  return pairs;
}

// dlib solves over integers; costs are at most the gate (about
// 11), so this scale keeps six decimal places of resolution.
constexpr double kCostScale = 1e6;

// Bonus every gated pairing earns on top of its cost, bigger
// than any total the scaled costs reach, so a match always wins.
constexpr long kPairReward = 1000000000000L;

// The optimal pairing, from dlib's Hungarian solver run on a
// zero-padded square copy of the rectangular gated grid.
std::vector<std::pair<int, int>> hungarian_pairs(
    const Eigen::MatrixXd& cost, double gate) {
  const long row_count = cost.rows();
  const long column_count = cost.cols();
  const long side = std::max(row_count, column_count);
  std::vector<std::pair<int, int>> pairs;
  if (side == 0) return pairs;

  // Padding rows and columns stay zero, so dlib never prefers
  // them over a real gated pairing's positive reward.
  dlib::matrix<long> reward(side, side);
  reward = 0;
  for (long row = 0; row < row_count; ++row)
    for (long column = 0; column < column_count; ++column)
      if (cost(row, column) <= gate)
        reward(row, column) =
            kPairReward -
            static_cast<long>(cost(row, column) * kCostScale);

  const std::vector<long> column_of_row =
      dlib::max_cost_assignment(reward);
  for (long row = 0; row < row_count; ++row) {
    const long column = column_of_row[row];
    if (column >= column_count || cost(row, column) > gate) continue;
    pairs.emplace_back(static_cast<int>(row),
                       static_cast<int>(column));
  }
  return pairs;
}

}  // namespace

// Runs the requested method, then derives which rows and columns
// were left unmatched; feeds the track life cycle in tracker.cpp.
Assignment assign(const Eigen::MatrixXd& cost, double gate,
                  AssignmentMethod method) {
  Assignment assignment;
  assignment.pairs = method == AssignmentMethod::Hungarian
                         ? hungarian_pairs(cost, gate)
                         : greedy_pairs(cost, gate);
  std::vector<bool> row_matched(cost.rows(), false),
      column_matched(cost.cols(), false);
  for (const auto& [row, column] : assignment.pairs) {
    row_matched[row] = true;
    column_matched[column] = true;
  }
  for (int row = 0; row < cost.rows(); ++row)
    if (!row_matched[row]) assignment.unmatched_rows.push_back(row);
  for (int column = 0; column < cost.cols(); ++column)
    if (!column_matched[column])
      assignment.unmatched_columns.push_back(column);
  return assignment;
}
