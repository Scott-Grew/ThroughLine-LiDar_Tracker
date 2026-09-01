#pragma once
#include <string>
#include <vector>
#include "log.hpp"
#include "types.hpp"
void export_tracks(const std::string& path, const SegmentLog& segment,
                   const std::vector<std::vector<Track>>&
                       confirmed_tracks_per_frame);
