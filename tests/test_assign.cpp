#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <limits>
#include <random>
#include <vector>
#include "assign.hpp"

// This file checks the Hungarian method against the only thing that can be trusted more than an
// algorithm: trying every possibility by hand. A small enough cost matrix can be solved by brute
// force in a few hundred thousand steps, so 200 random gated matrices are solved twice, once by
// the algorithm actually used in the tracker and once by exhaustive search, and the two totals are
// required to agree. Matching more pairs always beats a lower-cost solution with fewer pairs -
// that is what the tracker actually wants, since an unmatched track becomes a miss - so both
// solvers score a candidate solution the same way: pair count first, total cost second.

namespace {

double objective(int pair_count, double total_cost) {
  return -static_cast<double>(pair_count) * 1e6 + total_cost;
}

double best_brute_force_objective(const Eigen::MatrixXd& cost, double gate, int row, std::vector<bool>& column_used, int pair_count, double total_cost) {
  if (row == cost.rows()) return objective(pair_count, total_cost);

  double best = best_brute_force_objective(cost, gate, row + 1, column_used, pair_count, total_cost);
  for (int column = 0; column < cost.cols(); ++column) {
    if (column_used[column] || cost(row, column) > gate) continue;
    column_used[column] = true;
    const double candidate = best_brute_force_objective(cost, gate, row + 1, column_used, pair_count + 1, total_cost + cost(row, column));
    column_used[column] = false;
    best = std::min(best, candidate);
  }
  return best;
}

}

TEST_CASE("hungarian matches brute force on small gated matrices") {
  constexpr double kGate = 8.0;
  std::mt19937_64 generator(1);
  std::uniform_int_distribution<int> dimension_draw(1, 6);
  std::uniform_real_distribution<double> cost_draw(0.0, 10.0);
  std::uniform_real_distribution<double> unreachable_draw(0.0, 1.0);

  for (int trial = 0; trial < 200; ++trial) {
    const int row_count = dimension_draw(generator);
    const int column_count = dimension_draw(generator);
    Eigen::MatrixXd cost(row_count, column_count);
    for (int row = 0; row < row_count; ++row)
      for (int column = 0; column < column_count; ++column)
        cost(row, column) = unreachable_draw(generator) < 0.2 ? std::numeric_limits<double>::infinity() : cost_draw(generator);

    const Assignment assignment = assign(cost, kGate, AssignmentMethod::Hungarian);
    double hungarian_total_cost = 0.0;
    for (const auto& [row, column] : assignment.pairs) hungarian_total_cost += cost(row, column);
    const double hungarian_objective = objective(static_cast<int>(assignment.pairs.size()), hungarian_total_cost);

    std::vector<bool> column_used(column_count, false);
    const double brute_force_result = best_brute_force_objective(cost, kGate, 0, column_used, 0, 0.0);

    INFO("trial " << trial << " rows " << row_count << " columns " << column_count);
    REQUIRE(hungarian_objective == Catch::Approx(brute_force_result).margin(1e-6));
  }
}
