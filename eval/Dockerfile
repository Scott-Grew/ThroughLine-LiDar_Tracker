# Builds Waymo's compute_tracking_metrics_main from their repository and makes
# it the entrypoint; run_official.sh passes it the two Objects proto files.

FROM gcr.io/bazel-public/bazel:6.4.0

# Root, because the base image's default user cannot write into /workspace.
USER root

RUN git clone --depth 1 https://github.com/waymo-research/waymo-open-dataset /workspace/waymo-open-dataset

# Bazel runs from src/, where the clone's WORKSPACE file lives. Only the
# tracking metrics target is built; its dependencies do not include TensorFlow.
WORKDIR /workspace/waymo-open-dataset/src
RUN bazel build --jobs=2 //waymo_open_dataset/metrics/tools:compute_tracking_metrics_main

ENTRYPOINT ["/workspace/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main"]
