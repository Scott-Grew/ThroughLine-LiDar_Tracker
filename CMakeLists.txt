cmake_minimum_required(VERSION 3.28)
project(tracker CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wall -Wextra -Wpedantic)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(catch2)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(cli11)

FetchContent_Declare(
  dlib
  GIT_REPOSITORY https://github.com/davisking/dlib.git
  GIT_TAG v19.24.6
)
FetchContent_Populate(dlib)
add_library(dlib_headers INTERFACE)
target_include_directories(dlib_headers INTERFACE ${dlib_SOURCE_DIR})
target_compile_definitions(dlib_headers INTERFACE DLIB_NO_GUI_SUPPORT)

add_library(tracker_core
  src/log.cpp
  src/perturb.cpp
  src/filter.cpp
  src/assign.cpp
  src/tracker.cpp
  src/predict.cpp
  src/replay.cpp
  src/export.cpp
)
target_include_directories(tracker_core PUBLIC src)
target_link_libraries(tracker_core PUBLIC Eigen3::Eigen dlib_headers)

# Foxglove SDK 0.27.0 for x86_64 macOS: a prebuilt C library plus C++ wrapper
# sources with their own package config, so it is populated and then found.
FetchContent_Declare(
  foxglove_sdk
  URL "https://github.com/foxglove/foxglove-sdk/releases/download/\
sdk/v0.27.0/foxglove-v0.27.0-cpp-x86_64-apple-darwin.zip"
)
FetchContent_Populate(foxglove_sdk)
list(APPEND CMAKE_PREFIX_PATH "${foxglove_sdk_SOURCE_DIR}")
find_package(foxglove-sdk CONFIG REQUIRED)
foxglove_sdk_add_cpp_library(foxglove_cpp TYPE STATIC REMOTE_ACCESS OFF)

# The wrapper's static archive leaves zstd and lz4 unresolved; both come from
# Homebrew under /usr/local.
find_library(ZSTD_LIBRARY NAMES zstd PATHS /usr/local/lib)
find_library(LZ4_LIBRARY NAMES lz4 PATHS /usr/local/lib)
if(NOT ZSTD_LIBRARY OR NOT LZ4_LIBRARY)
  message(FATAL_ERROR "foxglove_cpp needs zstd and lz4 (brew install zstd lz4)")
endif()
target_link_libraries(foxglove_cpp PUBLIC
  ${ZSTD_LIBRARY}
  ${LZ4_LIBRARY}
  "-framework IOKit"
  "-framework CoreFoundation"
  "-framework Security"
  "-framework SystemConfiguration"
)

add_executable(tracker
  src/main.cpp
  src/recording.cpp
)
target_link_libraries(tracker PRIVATE tracker_core foxglove_cpp CLI11::CLI11)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(tracker_tests
  tests/test_determinism.cpp
  tests/test_filter.cpp
  tests/test_assign.cpp
  tests/test_allocation.cpp
)
target_link_libraries(tracker_tests PRIVATE tracker_core Catch2::Catch2WithMain)
catch_discover_tests(tracker_tests)
