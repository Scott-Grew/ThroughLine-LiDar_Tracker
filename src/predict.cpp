#include "predict.hpp"

ConstantTurnRatePredictor::ConstantTurnRatePredictor(FilterNoise noise) : noise_(noise) {}

PredictedPath ConstantTurnRatePredictor::predict(const Track& track, double horizon_seconds, double step_seconds) const {
  (void)track;
  (void)horizon_seconds;
  (void)step_seconds;
  return PredictedPath{};
}
