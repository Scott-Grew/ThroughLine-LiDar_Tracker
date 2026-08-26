# This image builds and serves Waymo's own tracking evaluator. It exists because Scott's rule for
# this project is that nothing we write may compute a score - only Waymo's binary is allowed to
# produce the number that counts. It clones the waymo-open-dataset repository fresh and compiles
# just compute_tracking_metrics_main with bazel, so the binary is byte-for-byte the one Waymo
# ships, then points the entrypoint at it so run_official.sh has something to run the two Objects
# proto files through. Nothing in this file decides a metric; it only builds the tool that does.
#
# This stays a single stage instead of copying the built binary into a slim ubuntu:22.04 runtime
# image: in this build environment, pulls of docker.io/library/ubuntu and docker.io/library/alpine
# hung indefinitely on every attempt (verified over four separate tries), while gcr.io pulls -
# including this same 1 GB base image - completed without issue. Reusing the already-reachable
# builder image as the runtime image trades a smaller final image for not depending on a registry
# that this network will not reliably reach.

FROM gcr.io/bazel-public/bazel:6.4.0

# The base image's entrypoint is bazel itself; that only matters for `docker run` with the image
# left as-is, and is overridden below. It has no bearing on the RUN steps, which execute through
# the shell regardless of entrypoint. Root is needed because the image's default user cannot write
# into /workspace.
USER root

RUN git clone --depth 1 https://github.com/waymo-research/waymo-open-dataset /workspace/waymo-open-dataset

# WORKSPACE lives in the src/ subdirectory of the clone, so bazel has to be invoked from there.
# Only the tracking metrics binary is built - not TensorFlow, not the python package, not anything
# else the workspace can reach - because that target's own dependency chain (label proto, the
# metrics/tracking_metrics library, absl, glog) never touches TensorFlow, and pulling more than
# that in would fight the CPU and disk budget this project runs under for no reason.
WORKDIR /workspace/waymo-open-dataset/src
RUN bazel build --jobs=2 //waymo_open_dataset/metrics/tools:compute_tracking_metrics_main

ENTRYPOINT ["/workspace/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main"]
