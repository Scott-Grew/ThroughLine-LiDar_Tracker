import argparse


def read_components(root, segment):
    raise NotImplementedError


def range_image_to_points(range_image, calibration):
    raise NotImplementedError


def measure_box_jitter(boxes_by_frame):
    raise NotImplementedError


def write_log(path, frames):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--out", required=True)
    parser.parse_args()


if __name__ == "__main__":
    main()
