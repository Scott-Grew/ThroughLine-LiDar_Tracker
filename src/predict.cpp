#include "predict.hpp"

#include <cmath>

// This file answers "if nothing else is ever seen again, where does this track go next." It never
// touches a real track's state, only a disposable copy, so running a prediction can never change
// what the tracker itself believes. The viewer calls this once per track every frame to draw the
// dotted path ahead of each object and the growing uncertainty ellipse around it; nothing in the
// tracker or the metrics depends on this file at all.

ConstantTurnRatePredictor::ConstantTurnRatePredictor(FilterNoise noise) : noise_(noise) {}

// Coasts a track's filter state forward in fixed steps out to the requested horizon, using the
// same constant-turn-rate motion the tracker predicts with between real detections, and records
// where the object would be and how sure of that the filter would be at every step. Because each
// step widens the uncertainty on top of the last, the path drawn furthest out is always the least
// certain one.
PredictedPath ConstantTurnRatePredictor::predict(const Track& track, double horizon_seconds, double step_seconds) const {
  TrackState state = track.state;
  const int step_count = static_cast<int>(std::round(horizon_seconds / step_seconds));

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
