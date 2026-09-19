#pragma once
#include <string>
#include <vector>
#include "log.hpp"
#include "types.hpp"

// Declares export_tracks, which writes confirmed tracks to a CSV
// for Waymo's submission tooling. See export.cpp for the format.

void export_tracks(const std::string& path, const SegmentLog& segment,
                   const std::vector<std::vector<Track>>&
                       confirmed_tracks_per_frame);
