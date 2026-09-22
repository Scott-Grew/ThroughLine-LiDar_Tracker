// Writes a finished replay to an .mcap file for playback in
// Lichtblick.

#pragma once
#include <string>
#include <vector>
#include "log.hpp"
#include "predict.hpp"
#include "types.hpp"

void save_replay_recording(
    const std::string& path, const SegmentLog& segment,
    const std::vector<std::vector<Track>>& confirmed_tracks_per_frame,
    const ConstantTurnRatePredictor& predictor, double horizon_seconds,
    double step_seconds);
