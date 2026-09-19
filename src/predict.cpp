#include "predict.hpp"

#include <cmath>

// Predicts a track's future path for the recording to draw; nothing
// in the tracker depends on this file or the copy it predicts from.

// Stores the process noise applied while coasting a track's state.
ConstantTurnRatePredictor::ConstantTurnRatePredictor(
    FilterNoise noise)
    : noise_(noise) {}

// Coasts a copy of the track's state forward in fixed steps out to
// the horizon, widening its uncertainty with each simulated step.
PredictedPath ConstantTurnRatePredictor::predict(
    const Track& track, double horizon_seconds,
    double step_seconds) const {
  TrackState state = track.state;
  const int step_count =
      static_cast<int>(std::round(horizon_seconds / step_seconds));

  PredictedPath path;
  path.positions.reserve(step_count);
  path.covariances.reserve(step_count);
  for (int step = 1; step <= step_count; ++step) {
    ::predict(state, step_seconds, noise_);
    path.positions.emplace_back(state.mean(0), state.mean(1));
    path.covariances.push_back(state.covariance.block<2, 2>(0, 0));
  }
  return path;
}
