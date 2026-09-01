#pragma once
#include <string>
#include <vector>
#include "log.hpp"
#include "predict.hpp"
#include "types.hpp"

// This file is the only place the project draws anything.
// save_replay_recording writes a finished replay's tracks,
// predictions and lidar points into an .mcap file for playback in
// Lichtblick. Nothing outside recording.cpp knows what a Foxglove
// message looks like.

void save_replay_recording(
    const std::string& path, const SegmentLog& segment,
    const std::vector<std::vector<Track>>& confirmed_tracks_per_frame,
    const ConstantTurnRatePredictor& predictor,
    double horizon_seconds, double step_seconds);
