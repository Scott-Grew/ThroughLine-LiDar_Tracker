#pragma once
#include <vector>
#include <Eigen/Dense>
#include "filter.hpp"
#include "types.hpp"

// Declares the constant-turn-rate predictor the recording uses to
// coast a track's future path; predict.cpp holds the definitions.

// One track's coasted future: a world-frame xy position and
// covariance for each step out to the horizon.
struct PredictedPath {
  std::vector<Eigen::Vector2d> positions;
  std::vector<Eigen::Matrix2d> covariances;
};

// Coasts a track's filter state forward under a constant-turn-rate
// motion model, without ever touching the tracker's real track.
class ConstantTurnRatePredictor {
 public:
  explicit ConstantTurnRatePredictor(FilterNoise noise);
  PredictedPath predict(const Track& track, double horizon_seconds,
                        double step_seconds) const;

 private:
  FilterNoise noise_;
};
