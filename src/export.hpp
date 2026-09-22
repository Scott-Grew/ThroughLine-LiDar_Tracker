// Declares export_tracks, which writes confirmed tracks to the
// CSV the scripts in eval/ score; export.cpp holds the columns.

#pragma once
#include <string>
#include <vector>
#include "log.hpp"
#include "types.hpp"

// Writes one CSV row per confirmed track per frame to path, in vehicle frame.
void export_tracks(
    const std::string& path, const SegmentLog& segment,
    const std::vector<std::vector<Track>>& confirmed_tracks_per_frame);
