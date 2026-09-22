// Predicts a track's future path for the recording to draw, from a
// copy of the track's filter state.

#include "predict.hpp"

#include <cmath>

ConstantTurnRatePredictor::ConstantTurnRatePredictor(FilterNoise noise)
    : noise_(noise) {}

PredictedPath ConstantTurnRatePredictor::predict(const Track& track,
                                                 double horizon_seconds,
                                                 double step_seconds) const {
  TrackState state = track.state;
  const int step_count =
      static_cast<int>(std::round(horizon_seconds / step_seconds));

  PredictedPath path;
  path.positions.reserve(step_count);
  path.covariances.reserve(step_count);
  for (int step = 1; step <= step_count; ++step) {
    ::predict(state, step_seconds, noise_);
    path.positions.emplace_back(state.mean(kPositionXIndex),
                                state.mean(kPositionYIndex));
    path.covariances.push_back(
        state.covariance.block<2, 2>(kPositionXIndex, kPositionXIndex));
  }
  return path;
}
