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

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.5
)
FetchContent_MakeAvailable(imgui)

find_package(OpenGL REQUIRED)

add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

add_library(tracker_core
  src/log.cpp
  src/perturb.cpp
  src/filter.cpp
  src/assign.cpp
  src/tracker.cpp
  src/iou.cpp
  src/metrics.cpp
  src/predict.cpp
  src/replay.cpp
  src/export.cpp
)
target_include_directories(tracker_core PUBLIC src)
target_link_libraries(tracker_core PUBLIC Eigen3::Eigen)

# Pinned to 0.26.0: Rerun stopped shipping a macOS x86_64 prebuilt library after this release, and
# this machine is an Intel Mac, so 0.26.0 is the newest version that still links here.
FetchContent_Declare(
  rerun_sdk
  URL https://github.com/rerun-io/rerun/releases/download/0.26.0/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

add_executable(tracker
  src/main.cpp
  src/viewer.cpp
)
target_link_libraries(tracker PRIVATE tracker_core imgui rerun_sdk)
target_compile_definitions(tracker PRIVATE GL_SILENCE_DEPRECATION)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

add_executable(tracker_tests
  tests/test_determinism.cpp
  tests/test_filter.cpp
  tests/test_assign.cpp
  tests/test_iou.cpp
  tests/test_metrics.cpp
  tests/test_allocation.cpp
)
target_link_libraries(tracker_tests PRIVATE tracker_core Catch2::Catch2WithMain)
catch_discover_tests(tracker_tests)
