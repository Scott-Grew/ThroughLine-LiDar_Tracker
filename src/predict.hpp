#pragma once
#include <vector>
#include <Eigen/Dense>
#include "filter.hpp"
#include "types.hpp"

struct PredictedPath {
  std::vector<Eigen::Vector2d> positions;
  std::vector<Eigen::Matrix2d> covariances;
};

class ConstantTurnRatePredictor {
 public:
  explicit ConstantTurnRatePredictor(FilterNoise noise);
  PredictedPath predict(const Track& track, double horizon_seconds,
                        double step_seconds) const;

 private:
  FilterNoise noise_;
};
