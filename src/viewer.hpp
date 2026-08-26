#pragma once
#include <string>
#include <vector>
#include "log.hpp"
#include "predict.hpp"
#include "replay.hpp"

// This file is the only doorway into drawing anything. run_viewer opens the small ImGui sliders
// window while a replay runs on another thread, reading whatever the replay just published;
// save_replay_recording walks a finished replay's stored tracks back through the same drawing
// code to leave behind a gapless .mcap file for later playback in Lichtblick. Nothing outside
// viewer.cpp knows what a Foxglove message or an ImGui widget looks like - this header is the
// entire surface the rest of the project sees.

int run_viewer(SnapshotExchange& exchange, LiveControls& controls);
void save_replay_recording(const std::string& path, const SegmentLog& segment,
                           const std::vector<std::vector<Track>>& confirmed_tracks_per_frame,
                           const Predictor& predictor, double horizon_seconds, double step_seconds);
