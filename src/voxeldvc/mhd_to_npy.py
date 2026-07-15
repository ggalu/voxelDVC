# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-06-14 22:36:02
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-14 22:37:45
#!/usr/bin/env python3
"""Convert .mhd/.raw (MetaImage) files to plain .npy numpy arrays."""

import argparse
import gzip
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Map MetaImage element types to numpy dtypes
MET_TO_NUMPY = {
    "MET_CHAR": np.int8,
    "MET_UCHAR": np.uint8,
    "MET_SHORT": np.int16,
    "MET_USHORT": np.uint16,
    "MET_INT": np.int32,
    "MET_UINT": np.uint32,
    "MET_LONG": np.int64,
    "MET_ULONG": np.uint64,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


def read_mhd(mhd_path):
    """Parse a .mhd header file into a dict of key -> value strings."""
    header = {}
    with open(mhd_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def mhd_to_array(mhd_path):
    """Load a .mhd/.raw pair and return a numpy array with axes ordered (z, y, x, ...)."""
    header = read_mhd(mhd_path)

    dim_size = [int(v) for v in header["DimSize"].split()]
    element_type = header["ElementType"]
    if element_type not in MET_TO_NUMPY:
        raise ValueError(f"Unsupported ElementType: {element_type}")
    dtype = np.dtype(MET_TO_NUMPY[element_type])

    if header.get("BinaryDataByteOrderMSB", "False").lower() == "true":
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")

    data_file = header.get("ElementDataFile", "").strip()
    if not data_file or data_file == "LOCAL":
        raise ValueError("Inline (LOCAL) data is not supported")

    raw_path = os.path.join(os.path.dirname(mhd_path), data_file)
    compressed = header.get("CompressedData", "False").lower() == "true"

    if compressed:
        with gzip.open(raw_path, "rb") as f:
            raw_data = f.read()
        array = np.frombuffer(raw_data, dtype=dtype)
    else:
        array = np.fromfile(raw_path, dtype=dtype)

    # MetaImage stores DimSize as (x, y, z, ...); numpy shape is reversed (..., z, y, x)
    array = array.reshape(dim_size[::-1])
    return array, header


def plot_histogram(array, title, out_path, bins=256):
    """Plot and save a histogram of detector count values, excluding the 0 (background) bin."""
    nonzero = array[array != 0]

    fig, ax = plt.subplots()
    ax.hist(nonzero.ravel(), bins=bins)
    ax.set_xlabel("Detector counts")
    ax.set_ylabel("Voxel count")
    ax.set_title(title)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.show()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert .mhd/.raw files to .npy")
    parser.add_argument("mhd_files", nargs="+", help="Path(s) to .mhd file(s)")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory to write .npy files to (default: same directory as input)",
    )
    parser.add_argument(
        "--histogram", action="store_true",
        help="Also plot a histogram of the voxel values (excluding 0/background) and save as .png",
    )
    parser.add_argument(
        "--bins", type=int, default=256,
        help="Number of histogram bins (default: 256)",
    )
    args = parser.parse_args(argv)

    for mhd_path in args.mhd_files:
        print(f"Converting {mhd_path} ...")
        array, header = mhd_to_array(mhd_path)

        out_dir = args.output_dir or os.path.dirname(mhd_path)
        base_name = os.path.splitext(os.path.basename(mhd_path))[0]
        out_path = os.path.join(out_dir, base_name + ".npy")

        np.save(out_path, array)
        print(f"  shape={array.shape} dtype={array.dtype} -> {out_path}")

        if args.histogram:
            hist_path = os.path.join(out_dir, base_name + "_histogram.png")
            plot_histogram(array, base_name, hist_path, bins=args.bins)
            print(f"  histogram -> {hist_path}")


if __name__ == "__main__":
    sys.exit(main())
