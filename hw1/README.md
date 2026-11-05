# Homework 1 — Image Enhancement

## Overview
This repository contains a simple image processing pipeline. 
The goal is to test classical enhancement methods on images suffering from low contrast, non-uniform illumination, and noise. 

To evaluate the methods quantitatively, I synthetically degraded clean original images (by reducing contrast and adding Gaussian noise) so the originals could serve as a clean reference for computing PSNR and SSIM metrics.

## Images and source

| Filename | Subject | Main characteristic |
| --- | --- | --- |
| `mare.jpeg` | Sea landscape | Low contrast in some shaded regions |
| `alba.jpeg` | Sunrise landscape | Non-uniform illumination and dark foreground |
| `acquario.jpeg` | Aquarium | Poorly visible details under blue lighting |

*Note: These are personal photographs by the author, used with permission.*

## Processing pipeline

For every original image, `main.py`:

1. Produces a synthetic degradation using a contrast-reducing affine transformation and Gaussian noise.
2. Applies three enhancement algorithms **independently to the same degraded image**.
3. Saves degraded and enhanced PNG files.
4. Generates a three-panel visual comparison for each method(original, degraded, and the three enhanced outputs).
5. Computes PSNR and SSIM relative to the original, saving results to `results/metrics.csv`.

### Degradation parameters

\[
I_d = \operatorname{clip}(0.6 I_o + 30 + N, 0, 255), \quad N\sim \mathcal{N}(0,5^2)
\]

- Gain: `0.6`
- Bias: `30`
- Gaussian noise mean: `0`
- Gaussian noise standard deviation: `5` (in 8-bit intensity units)
- NumPy random seed: `42`
- Order of images: `mare`, `alba`, `acquario`
- Conversion: computation in floating point, clipping to `[0,255]`, then conversion to `uint8`

The fixed random seed and image order make noise generation reproducible for the same input files and software environment.

### Methods

- **Baseline:** degraded image without enhancement.
- **Linear Contrast Stretching:** global min-max intensity rescaling to `[0,255]` on the HSV V channel; handles constant V by returning the input. This is a specific application of the lecture's `g(x) = af(x) + b` (bias and gain) transformation.
- **Histogram Equalization:** global histogram equalization (`cv2.equalizeHist`) of the HSV V channel.
- **Gamma Correction:** `V_out = 255 * (V_in / 255)^(1/gamma)` on the HSV V channel, using `gamma = 2.2`.

These methods may accentuate noise and change the visual appearance of colors. The same parameters are used for all three photographs.

### Evaluation

PSNR (in dB) and SSIM are computed with `scikit-image` against the **original** color photographs. Both are full-reference metrics; visual naturalness is evaluated separately. Results are listed per image and method in the CSV file.

## Setup (Windows / PowerShell)

From the project root, create a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, you can run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current shell (if permitted) or call `.\.venv\Scripts\python.exe` directly instead of `python`.

Install the dependencies:

```powershell
python -m pip install opencv-python numpy scikit-image matplotlib
```

## Setup (macOS / Terminal)

From the project root, create a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install opencv-python numpy scikit-image matplotlib
```

## Run the complete experiment

Ensure the three original files are in `images/original/`, then run **one command** from the project root for Windows:

```powershell
python main.py
```
or for macOs:

```bash
python3.12 main.py
```

This regenerates all degraded images, processed images, comparisons, and `results/metrics.csv`.

The individual scripts remain available if you prefer to run one method at a time:

```powershell
python degradation.py
python test_enhancement.py
python histogram.py
python gamma.py
python evaluate.py
```

These are **optional**: `main.py` runs the full experiment independently without executing the other script files.

## Directory structure

```text
HW1_VR545790/
├── README.md
├── requirements.txt
├── main.py
├── enhancement.py              # All enhancement and metric functions
├── degradation.py
├── test_enhancement.py
├── histogram.py
├── gamma.py
├── evaluate.py
├── images/
│   ├── original/
│   │   ├── mare.jpeg
│   │   ├── alba.jpeg
│   │   └── acquario.jpeg
│   └── degraded/
│       ├── mare.png
│       ├── alba.png
│       └── acquario.png
└── results/
    ├── processed/              # *_linear.png, *_histogram.png, *_gamma.png
    ├── comparisons/            # *_all_methods.png (+ optional individual comparisons)
    └── metrics.csv
```


## References

- Chapters 03 (*Digital Images*) and 04 (*Image Processing*).
- [OpenCV documentation](https://docs.opencv.org/)
- [scikit-image documentation](https://scikit-image.org/docs/stable/)

