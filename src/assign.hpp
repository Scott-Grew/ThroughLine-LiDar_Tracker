#pragma once
#include <vector>
#include <Eigen/Dense>

enum class AssignmentMethod { Hungarian, Greedy };

struct Assignment {
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_columns;
};

Assignment assign(const Eigen::MatrixXd& cost, double gate, AssignmentMethod method);
