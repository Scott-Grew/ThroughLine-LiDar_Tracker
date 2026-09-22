// Declares the constant-turn-rate predictor the recording uses to
// forecast a track's future path; predict.cpp holds the definitions.

#pragma once
#include <vector>
#include <Eigen/Dense>
#include "filter.hpp"
#include "types.hpp"

// One track's forecast path, a world-frame xy position and covariance
// for each step out to the horizon.
struct PredictedPath {
  std::vector<Eigen::Vector2d> positions;
  std::vector<Eigen::Matrix2d> covariances;
};

// Forecasts a copy of a track's filter state forward under the
// constant-turn-rate motion model.
class ConstantTurnRatePredictor {
 public:
  explicit ConstantTurnRatePredictor(FilterNoise noise);
  PredictedPath predict(const Track& track, double horizon_seconds,
                        double step_seconds) const;

 private:
  FilterNoise noise_;
};
