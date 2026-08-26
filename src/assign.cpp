#include "assign.hpp"

#include <algorithm>
#include <limits>
#include <vector>

// This file answers one question: given a grid of costs between tracks and detections, who goes
// with whom. The tracker builds one such grid per object class per frame and hands it here; this
// file hands back a list of matched pairs plus whatever was left over on each side. Two ways of
// answering are offered - a fast greedy pass that grabs the cheapest pairing first, and the
// Hungarian algorithm, which is slower but finds the assignment with the lowest total cost across
// the whole grid. Which one runs is a setting on the tracker, not a decision made in here. Nothing
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

// A cost stood in for anything past the gate, large enough that the optimal search will never
// choose it over a real pairing but not so large that it breaks the arithmetic below.
constexpr double kUnreachable = 1e9;

// The optimal answer: the Jonker-Volgenant form of the Hungarian algorithm, which finds the set
// of pairings with the lowest possible total cost rather than settling for the first cheap one
// found. It works on a square problem internally, so a wide grid is transposed to a tall one
// first and the pairs are flipped back before returning. Costs past the gate are replaced with
// the unreachable value above so the algorithm can still run its bookkeeping over the whole grid
// without ever actually choosing a pairing that should have been refused.
std::vector<std::pair<int, int>> hungarian_pairs(const Eigen::MatrixXd& cost, double gate) {
  const bool transposed = cost.rows() > cost.cols();
  const Eigen::MatrixXd matrix = transposed ? Eigen::MatrixXd(cost.transpose()) : cost;
  const int row_count = static_cast<int>(matrix.rows());
  const int column_count = static_cast<int>(matrix.cols());

  std::vector<double> row_offset(row_count + 1, 0.0), column_offset(column_count + 1, 0.0);
  std::vector<int> row_of_column(column_count + 1, 0), previous_column(column_count + 1, 0);

  for (int row = 1; row <= row_count; ++row) {
    row_of_column[0] = row;
    int current_column = 0;
    std::vector<double> reduced_minimum(column_count + 1, std::numeric_limits<double>::infinity());
    std::vector<bool> visited(column_count + 1, false);
    do {
      visited[current_column] = true;
      const int current_row = row_of_column[current_column];
      double delta = std::numeric_limits<double>::infinity();
      int next_column = 0;
      for (int column = 1; column <= column_count; ++column) {
        if (visited[column]) continue;
        const double entry = matrix(current_row - 1, column - 1) <= gate ? matrix(current_row - 1, column - 1) : kUnreachable;
        const double reduced = entry - row_offset[current_row] - column_offset[column];
        if (reduced < reduced_minimum[column]) {
          reduced_minimum[column] = reduced;
          previous_column[column] = current_column;
        }
        if (reduced_minimum[column] < delta) {
          delta = reduced_minimum[column];
          next_column = column;
        }
      }
      for (int column = 0; column <= column_count; ++column) {
        if (visited[column]) {
          row_offset[row_of_column[column]] += delta;
          column_offset[column] -= delta;
        } else {
          reduced_minimum[column] -= delta;
        }
      }
      current_column = next_column;
    } while (row_of_column[current_column] != 0);
    do {
      const int column = previous_column[current_column];
      row_of_column[current_column] = row_of_column[column];
      current_column = column;
    } while (current_column != 0);
  }

  std::vector<std::pair<int, int>> pairs;
  for (int column = 1; column <= column_count; ++column) {
    const int row = row_of_column[column];
    if (row == 0) continue;
    const double real_cost = matrix(row - 1, column - 1);
    if (real_cost > gate) continue;
    if (transposed) pairs.emplace_back(column - 1, row - 1);
    else pairs.emplace_back(row - 1, column - 1);
  }
  std::sort(pairs.begin(), pairs.end());
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
