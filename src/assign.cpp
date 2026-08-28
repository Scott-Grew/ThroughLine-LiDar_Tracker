#include "assign.hpp"

#include <algorithm>
#include <vector>
#include <dlib/optimization/max_cost_assignment.h>

// This file answers one question: given a grid of costs between tracks and detections, who goes
// with whom. The tracker builds one such grid per object class per frame and hands it here; this
// file hands back a list of matched pairs plus whatever was left over on each side. Two ways of
// answering are offered - a fast greedy pass that grabs the cheapest pairing first, and dlib's
// Hungarian solver, which finds the assignment with the lowest total cost across the whole grid. Which one runs is a setting on the tracker, not a decision made in here. Nothing
// downstream cares how the answer was reached, only which boxes ended up paired.

namespace {

// One possible pairing and what it would cost, used only while the greedy method is picking
// through every option in order from cheapest to most expensive.
struct Candidate {
  double cost;
  int row;
  int column;
};

// The fast, approximate answer: sort every pairing under the gate from cheapest to most
// expensive, then walk that list taking each pairing whose row and column are both still free.
// This can miss a lower-cost arrangement that the optimal method would have found, because an
// early cheap pairing can block a row or column that a later, better combination needed.
std::vector<std::pair<int, int>> greedy_pairs(const Eigen::MatrixXd& cost, double gate) {
  std::vector<Candidate> candidates;
  for (int row = 0; row < cost.rows(); ++row)
    for (int column = 0; column < cost.cols(); ++column)
      if (cost(row, column) <= gate) candidates.push_back({cost(row, column), row, column});
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& first, const Candidate& second) {
    if (first.cost != second.cost) return first.cost < second.cost;
    if (first.row != second.row) return first.row < second.row;
    return first.column < second.column;
  });
  std::vector<bool> row_used(cost.rows(), false), column_used(cost.cols(), false);
  std::vector<std::pair<int, int>> pairs;
  for (const Candidate& candidate : candidates) {
    if (row_used[candidate.row] || column_used[candidate.column]) continue;
    row_used[candidate.row] = true;
    column_used[candidate.column] = true;
    pairs.emplace_back(candidate.row, candidate.column);
  }
  return pairs;
}

// Scale applied to costs before handing them to dlib, which solves over integers. Costs are at
// most the gate (about 11), so a scale of one million keeps six decimal places of resolution.
constexpr double kCostScale = 1e6;

// A reward every pairing under the gate earns on top of its cost, larger than any total the
// scaled costs can reach, so the solver always prefers one more matched pair over any saving in
// cost. An unmatched track becomes a miss, which is what the tracker most wants to avoid.
constexpr long kPairReward = 1000000000000L;

// The optimal answer, from dlib's max_cost_assignment (the Hungarian method). dlib wants a
// square matrix and maximises, so the rectangular gated cost grid is embedded in a square of
// zeros where every pairing under the gate scores the pair reward minus its scaled cost, and
// every pairing past the gate or in the padding scores zero. Pairs the solver lands on padding or
// gated cells are dropped afterwards.
std::vector<std::pair<int, int>> hungarian_pairs(const Eigen::MatrixXd& cost, double gate) {
  const long row_count = cost.rows();
  const long column_count = cost.cols();
  const long side = std::max(row_count, column_count);
  std::vector<std::pair<int, int>> pairs;
  if (side == 0) return pairs;

  dlib::matrix<long> reward(side, side);
  reward = 0;
  for (long row = 0; row < row_count; ++row)
    for (long column = 0; column < column_count; ++column)
      if (cost(row, column) <= gate) reward(row, column) = kPairReward - static_cast<long>(cost(row, column) * kCostScale);

  const std::vector<long> column_of_row = dlib::max_cost_assignment(reward);
  for (long row = 0; row < row_count; ++row) {
    const long column = column_of_row[row];
    if (column >= column_count || cost(row, column) > gate) continue;
    pairs.emplace_back(static_cast<int>(row), static_cast<int>(column));
  }
  return pairs;
}

}

// The entry point everything else calls. Runs whichever method was asked for, then works out from
// the resulting pairs which rows and columns never got matched at all. The tracker treats an
// unmatched track as a possible miss and an unmatched detection as a possible new object, so this
// bookkeeping is what feeds the track life cycle in tracker.cpp.
Assignment assign(const Eigen::MatrixXd& cost, double gate, AssignmentMethod method) {
  Assignment assignment;
  assignment.pairs = method == AssignmentMethod::Hungarian ? hungarian_pairs(cost, gate) : greedy_pairs(cost, gate);
  std::vector<bool> row_matched(cost.rows(), false), column_matched(cost.cols(), false);
  for (const auto& [row, column] : assignment.pairs) {
    row_matched[row] = true;
    column_matched[column] = true;
  }
  for (int row = 0; row < cost.rows(); ++row)
    if (!row_matched[row]) assignment.unmatched_rows.push_back(row);
  for (int column = 0; column < cost.cols(); ++column)
    if (!column_matched[column]) assignment.unmatched_columns.push_back(column);
  return assignment;
}
