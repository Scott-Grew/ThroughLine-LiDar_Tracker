#pragma once
#include <vector>
#include <Eigen/Dense>

// Matches tracks to detections from a track-by-detection cost
// grid; called once per object class per frame by the tracker.

enum class AssignmentMethod { Hungarian, Greedy };

// The matched track/detection pairs from one assign() call, plus
// the rows and columns left unmatched.
struct Assignment {
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_columns;
};

Assignment assign(const Eigen::MatrixXd& cost, double gate,
                  AssignmentMethod method);
