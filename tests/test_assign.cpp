// Checks the Hungarian solver against exhaustive search on small random
// gated cost grids; both rank a solution by pair count first, then cost.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <limits>
#include <random>
#include <vector>
#include "assign.hpp"

namespace {

// Sizes of the randomised comparison and the weight that ranks pair count.
constexpr int kTrialCount = 200;
constexpr int kMaxGridSide = 6;
constexpr double kUnreachableFraction = 0.2;
constexpr double kGate = 8.0;
constexpr double kPairCountWeight = 1e6;

// Scores a solution so more pairs always wins and lower cost breaks ties;
// the 1e6 per pair exceeds any total cost these small grids can reach.
double objective(int pair_count, double total_cost) {
  return -static_cast<double>(pair_count) * kPairCountWeight + total_cost;
}

// Returns the best objective over all ways to pair rows from `row` onward
// with unused columns under the gate, by trying each one.
double best_brute_force_objective(const Eigen::MatrixXd& cost, double gate,
                                  int row, std::vector<bool>& column_used,
                                  int pair_count, double total_cost) {
  if (row == cost.rows()) return objective(pair_count, total_cost);

  double best_objective = best_brute_force_objective(
      cost, gate, row + 1, column_used, pair_count, total_cost);
  for (int column = 0; column < cost.cols(); ++column) {
    if (column_used[column] || cost(row, column) > gate) continue;
    column_used[column] = true;
    const double candidate_objective = best_brute_force_objective(
        cost, gate, row + 1, column_used, pair_count + 1,
        total_cost + cost(row, column));
    column_used[column] = false;
    best_objective = std::min(best_objective, candidate_objective);
  }
  return best_objective;
}

}  // namespace

// Solves 200 random grids of up to 6 by 6 both ways and requires equal
// objectives; a fifth of the cells are infinite to exercise the gate.
TEST_CASE("hungarian matches brute force") {
  std::mt19937_64 generator(1);
  std::uniform_int_distribution<int> dimension_draw(1, kMaxGridSide);
  std::uniform_real_distribution<double> cost_draw(0.0, 10.0);
  std::uniform_real_distribution<double> unreachable_draw(0.0, 1.0);

  for (int trial = 0; trial < kTrialCount; ++trial) {
    const int row_count = dimension_draw(generator);
    const int column_count = dimension_draw(generator);
    Eigen::MatrixXd cost(row_count, column_count);
    for (int row = 0; row < row_count; ++row)
      for (int column = 0; column < column_count; ++column)
        cost(row, column) = unreachable_draw(generator) < kUnreachableFraction
                                ? std::numeric_limits<double>::infinity()
                                : cost_draw(generator);

    const Assignment assignment =
        assign(cost, kGate, AssignmentMethod::Hungarian);
    double hungarian_total_cost = 0.0;
    for (const auto& [row, column] : assignment.pairs)
      hungarian_total_cost += cost(row, column);
    const double hungarian_objective = objective(
        static_cast<int>(assignment.pairs.size()), hungarian_total_cost);

    std::vector<bool> column_used(column_count, false);
    const double brute_force_result =
        best_brute_force_objective(cost, kGate, 0, column_used, 0, 0.0);

    INFO("trial " << trial << " rows " << row_count << " columns "
                  << column_count);
    REQUIRE(hungarian_objective ==
            Catch::Approx(brute_force_result).margin(1e-6));
  }
}
