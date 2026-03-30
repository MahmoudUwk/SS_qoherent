"""
Download MathWorks' Spectrum Sensing dataset and convert it to the
single HDF5 format expected by the spectrogram-segmentation notebook.

Corrects a ~36-column spatial misalignment present in the source data
between HDF4 labels and PNG spectrograms.

Usage:
    python3 download_dataset.py
"""

import hashlib
import os
import re
import sys
import tarfile

import h5py
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

MATHWORKS_URL = (
    "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/"
    "SpectrumSenseTrainingDataNetwork.tar.gz"
)

HDF4_LABEL_OFFSET = 258
HDF4_PIXEL_SIZE = 256 * 256
LABEL_MAP = {0: 0, 127: 1, 255: 2}
MASK_SHIFT = -36


def download_tar(tar_path):
    if os.path.exists(tar_path):
        print(f"Using existing archive: {tar_path}")
        return
    print(f"Downloading {MATHWORKS_URL}")
    n_bytes = int(requests.head(MATHWORKS_URL).headers.get("Content-Length", 0))
    with (
        requests.get(MATHWORKS_URL, stream=True, timeout=3) as r,
        open(tar_path, "wb") as out_file,
        tqdm(desc="Downloading", total=n_bytes, unit="B", unit_scale=True) as pbar,
    ):
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                out_file.write(chunk)
                pbar.update(len(chunk))


def parse_frame_info(member_name):
    match = re.match(r"TrainingData/(LTE_NR/)?(\w+)_frame_(\d+)\.hdf$", member_name)
    if not match:
        return None, None, None
    is_combined = match.group(1) is not None
    signal_type = match.group(2)
    frame_idx = int(match.group(3))
    if is_combined:
        signal_type = "LTE_NR"
    return signal_type, frame_idx, member_name


def convert(tar_path, output_path):
    frames = []
    print(f"Reading archive: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        hdf_members = [m for m in members if m.name.endswith(".hdf")]
        png_members = {m.name: m for m in members if m.name.endswith(".png")}

        print(f"Found {len(hdf_members)} HDF frames and {len(png_members)} PNG images")

        for member in tqdm(hdf_members, desc="Converting"):
            info = parse_frame_info(member.name)
            if info[0] is None:
                continue
            signal_type, frame_idx, _ = info

            f = tar.extractfile(member)
            if f is None:
                continue
            hdf_data = f.read()

            pixel_labels = np.frombuffer(
                hdf_data, dtype=np.uint8, offset=HDF4_LABEL_OFFSET, count=HDF4_PIXEL_SIZE
            ).reshape(256, 256)
            mask = np.vectorize(LABEL_MAP.get)(pixel_labels).astype(np.uint8)
            mask = np.roll(mask, MASK_SHIFT, axis=1)
            if MASK_SHIFT < 0:
                mask[:, MASK_SHIFT:] = 0
            elif MASK_SHIFT > 0:
                mask[:, :MASK_SHIFT] = 0

            png_name = member.name.replace(".hdf", ".png")
            if png_name not in png_members:
                print(f"Warning: PNG not found for {member.name}")
                continue

            png_f = tar.extractfile(png_members[png_name])
            if png_f is None:
                continue
            image = np.array(Image.open(png_f)).astype(np.uint8)

            frames.append((signal_type, frame_idx, image, mask))

    frames.sort(key=lambda x: (x[0], x[1]))

    n = len(frames)
    print(f"Writing {n} frames to {output_path}")

    data = np.empty((n, 256, 256, 3), dtype=np.uint8)
    masks = np.empty((n, 256, 256), dtype=np.uint8)
    signal_types = []

    for i, (signal_type, _, image, mask) in enumerate(frames):
        data[i] = image
        masks[i] = mask
        signal_types.append(signal_type.encode("ascii"))

    dt = np.dtype([("signal_type", "S6")])
    metadata = np.array([tuple([s]) for s in signal_types], dtype=dt)

    about_dt = np.dtype([("author", "S64"), ("name", "S64")])
    about = np.array(
        [("MathWorks / Qoherent", "Spectrum Sensing Dataset v1.0")], dtype=about_dt
    )

    with h5py.File(output_path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("masks", data=masks)

        meta_group = f.create_group("metadata")
        meta_group.create_dataset("metadata", data=metadata)
        meta_group.create_dataset("about", data=about)

    sha256_hash = hashlib.sha256()
    with open(output_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    print(f"SHA256: {sha256_hash.hexdigest()}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    tar_path = os.path.join(project_root, "SpectrumSenseTrainingDataNetwork.tar.gz")
    hdf5_path = os.path.join(project_root, "spectrum_sensing_dataset.hdf5")

    if os.path.exists(hdf5_path):
        print(f"Dataset already exists: {hdf5_path}")
        sys.exit(0)

    download_tar(tar_path)
    convert(tar_path, hdf5_path)
    os.remove(tar_path)
    print(f"Cleaned up {tar_path}")
    print("Done!")
