# Lesion Insertion Toolbox

Repository: [github.com/qurit/lesion-insertion-toolbox](https://github.com/qurit/lesion-insertion-toolbox)

GUI application for inserting synthetic lesions into PET and SPECT data, re-reconstructing the modified data, and analysing quantitative image-quality metrics. Built on [PyTomography](https://github.com/qurit/PyTomography).

Developed at [Qurit Lab, BC Cancer](https://qurit.ca/) and the [University of British Columbia](https://www.ubc.ca/).

## Overview

The toolbox supports a full research workflow:

1. Setup — choose PET or SPECT and load input data.
2. Reconstruction — run an initial reconstruction to define the image grid and inspect anatomy.
3. Lesion Insertion — place lesions, queue reconstruction jobs, project lesions into raw data, and save results.
4. Data Analysis — compare scenarios (recovery coefficient, noise–bias, CNR, edge artifacts, Detection Stability Index).

Lesions are defined on the reconstructed image grid. During queue execution, they are forward-projected into PET listmode or SPECT projections, then reconstructed with user-selected algorithms.

## Supported data and reconstruction

### PET

| Item | Details |
|------|---------|
| Input | Listmode folder (`.BLF`) with correction files (`.h5`); optional CT folder |
| Beds | Single-bed and multi-bed listmode |
| Algorithms | OSEM, BSREM |
| Lesion insertion | Analytical forward projection |

### SPECT

| Item | Details |
|------|---------|
| Input | DICOM projection file or folder; CT folder for attenuation map |
| Beds | Single-bed and multi-bed projections |
| Algorithms | OSEM, OSMAPOSL, BSREM, KEM |
| Lesion insertion | Analytical or Monte Carlo ([SIMIND](https://www.simind.phy.uu.se/), separate install) |

## Lesion models

- Spheres — radius and intensity defined in the viewer; placed by clicking the image.
- Library lesions — import real lesion shapes from NIfTI masks via the Real Lesion Library; optional uniform intensity (shape-only) mode for recovery studies.

Multiple lesions can be added, edited, and saved before queueing.

## GUI features

- Multi-planar viewer (axial, sagittal, coronal) with crosshairs and overlay options (reconstruction, CT, lesion mask).
- CT resampled to the reconstruction grid for display and registration.
- Reconstruction queue — multiple named jobs with different algorithms, iterations, subsets, and priors.
- Optional save after each iteration for convergence analysis.
- Real lesion library (`LesionLibrary.json`) — see below.

## Using LesionLibrary.json

The default library file is `LesionLibrary.json` in the repository root (also on [GitHub](https://github.com/qurit/lesion-insertion-toolbox)). It contains pre-built real lesion shapes.

1. Setup tab — load PET or SPECT data.
2. Reconstruction tab — run an initial reconstruction.
3. Lesion Insertion tab — click Real Lesion Library.
4. In the library window, under Configuration, click Load Library and select `LesionLibrary.json`.
5. In Lesion Library, click a lesion to preview it. Set Intensity Scale, Size Scale, and other options as needed.
6. Click Insert Lesion. The lesion is placed at the current crosshair position on the Lesion Insertion tab.

## Data Analysis tab

Load saved reconstruction results and compare scenarios across lesions:

- Recovery coefficient (RC) and bias
- Noise (CV) and noise–bias curves
- Contrast-to-noise ratio (CNR)
- Edge artifacts
- Detection Stability Index (DSI) — Radiomics mode; requires `pyradiomics`

## Requirements

NVIDIA GPU required (CUDA 12.4). Tested on Windows with Anaconda, Python 3.11.9, PyTorch 2.4.0, PyTomography 3.4.0.

| Component | Version | Install |
|-----------|---------|---------|
| Python | 3.11.9 | conda |
| PyTorch | 2.4.0 (CUDA 12.4) | conda |
| PyTomography | 3.4.0 | pip |
| PyQt5 | 5.15.11 | pip |
| SimpleITK | 2.4.0 | pip |
| pydicom | 3.0.1 | pip |
| h5py | 3.11.0 | pip |
| pyqtgraph | 0.13.7 | pip |
| scipy | 1.14.0 | pip |
| numpy | 1.26.4 | pip |
| pyradiomics | 3.0.1 | pip (optional; radiomics panel) |

Install PyTorch with conda first; remaining packages are in `requirements.txt`.

## Installation

Open Anaconda Prompt (or any shell with `conda`), then:

```bash
git clone https://github.com/qurit/lesion-insertion-toolbox.git
cd lesion-insertion-toolbox

conda create -n lesion_toolbox python=3.11.9
conda activate lesion_toolbox
conda install pytorch=2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

Verify GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected: `2.4.0 12.4 True`

Run:

```bash
python main.py
```

## Sample data

Example PET and SPECT datasets are not stored on GitHub (imaging files are too large). They are hosted together on [Zenodo](https://zenodo.org/).

### Where to find the data

| What | Where |
|------|--------|
| Source code + `LesionLibrary.json` | [github.com/qurit/lesion-insertion-toolbox](https://github.com/qurit/lesion-insertion-toolbox) (clone or download) |
| PET + SPECT imaging data | Zenodo — [add DOI link when published] |
| Folder layout after download | Unzip into `Sample Data/` inside your cloned repo (see below) |

### After you download from Zenodo

Place the extracted folders so your project looks like this:

```text
lesion-insertion-toolbox/
├── main.py
├── LesionLibrary.json
├── Sample Data/
│   ├── README.md
│   ├── PET/
│   │   └── LIST/          # .BLF listmode + corrections_*.h5
│   └── SPECT/
│       ├── CT/            # CT DICOM slices
│       └── Projection/    # projection DICOM
```

Then in the app: Setup tab → choose PET or SPECT → browse to the matching folders under `Sample Data/`.

## Project structure

```text
lesion-insertion-toolbox/
├── main.py
├── LesionLibrary.json      # Default real-lesion library
├── requirements.txt
├── Sample Data/              # Download from Zenodo (see README)
│   └── README.md
└── src/
    ├── main_window.py      # Main GUI and workflow
    ├── analysis_tab.py     # Data Analysis tab
    ├── dialogs.py          # Lesion library and settings dialogs
    ├── recon_pet.py        # PET reconstruction threads
    ├── recon_spect.py      # SPECT reconstruction threads
    ├── widgets.py          # Image viewers
    ├── models.py           # Lesion and queue data structures
    └── utils.py            # Lesion geometry and helpers
```

## Citation

Publication forthcoming. If you use this toolbox, please also cite [PyTomography](https://github.com/qurit/PyTomography) when appropriate.

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2025 Narges Aghakhanolia, Qurit Lab, BC Cancer and The University of British Columbia.
