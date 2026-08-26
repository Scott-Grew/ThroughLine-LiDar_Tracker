#pragma once
#include <vector>
#include <Eigen/Dense>
#include "filter.hpp"
#include "types.hpp"

struct PredictedPath {
  std::vector<Eigen::Vector2d> positions;
  std::vector<Eigen::Matrix2d> covariances;
};

class Predictor {
 public:
  virtual ~Predictor() = default;
  virtual PredictedPath predict(const Track& track, double horizon_seconds, double step_seconds) const = 0;
};

class ConstantTurnRatePredictor : public Predictor {
 public:
  explicit ConstantTurnRatePredictor(FilterNoise noise);
  PredictedPath predict(const Track& track, double horizon_seconds, double step_seconds) const override;
 private:
  FilterNoise noise_;
};
