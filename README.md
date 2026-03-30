# Spectrogram Segmentation

Train a PyTorch/Lightning model to segment LTE and NR signals in RF spectrograms.

## Linux Quickstart

1. Clone the repo.
```bash
git clone https://github.com/MahmoudUwk/SS_qoherent.git
cd Spectrogram-Segmentation
```

2. Create and activate a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies.
```bash
pip install -r requirements.txt
```

4. Download and build the dataset.
```bash
python3 download_dataset.py
```

This downloads the MathWorks source archive, converts it into `spectrum_sensing_dataset.hdf5`, fixes the known mask alignment issue, and removes the tarball after conversion.

5. Open the notebook.
```bash
jupyter notebook spectrogram_segmentation.ipynb
```

## Notes

- The final dataset is `spectrum_sensing_dataset.hdf5` in the repo root.
- If it already exists, `download_dataset.py` exits without downloading again.
