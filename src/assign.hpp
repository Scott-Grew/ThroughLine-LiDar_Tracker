// Matches tracks to detections from a track-by-detection cost
// grid; called once per object class per frame by the tracker.

#pragma once
#include <vector>
#include <Eigen/Dense>

// Which solver assign() runs: optimal Hungarian or cheapest-first.
enum class AssignmentMethod { Hungarian, Greedy };

// The matched track/detection pairs from one assign() call, plus
// the rows and columns left unmatched.
struct Assignment {
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_columns;
};

// Assignment of pairs under gate by method, with rows/columns left unmatched.
Assignment assign(const Eigen::MatrixXd& cost, double gate,
                  AssignmentMethod method);
