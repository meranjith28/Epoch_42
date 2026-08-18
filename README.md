# Epoch_42 — KLA Problem Statement: AI-Based Restoration of Degraded Images
### SEMICON India Hackathon 2026

## Overview
This repository restores degraded semiconductor-inspection images (speckle noise +
additive Gaussian noise + downsampling) back to their clean, full-resolution form,
using a NAFNet-style encoder-decoder with a PixelShuffle upsampling head.

## Setup

```bash
pip install -r requirements.txt
```

Requires an NVIDIA GPU with CUDA. No internet access, API keys, or additional model
downloads are needed at inference time — the checkpoint and config ship inside this
repository under `models/`.

## Running Inference

```bash
python run.py <input_dir> <output_dir>
```

- `<input_dir>`: a folder of degraded input images saved as `.npy` files.
- `<output_dir>`: created automatically if it doesn't exist; one restored `.npy` file
  is written per input file, using the same filename.

No other arguments are required. (Optional flags `--model_path`, `--config`, and
`--batch_size` exist for local debugging only and default to the bundled files.)

## Input / Output Contract

| | Format |
|---|---|
| Input | Grayscale `.npy` arrays, shape `(H, W)`. Values may extend slightly outside `[0,1]` (this is expected/intentional per the problem statement). |
| Output | Grayscale `.npy` arrays, shape `(H, W)`. Values clipped to `[0,1]`, no NaN/Inf. Resolution is the input resolution × the model's scale factor (2x). |

Filenames are preserved exactly between input and output.

## Model

- Architecture: NAFNet-style encoder-decoder (GroupNorm + SimpleGate + channel
  attention blocks) with a PixelShuffle-based 2x upsampling head.
- Trained for 50 epochs on paired GT/NoisyLR data (2,880 train / 320 val samples).
- Test-time augmentation: horizontal-flip averaging at inference.
- Target resolution is computed dynamically per input shape (not hardcoded), so the
  pipeline handles the dataset's mixed ~256×256 and ~512×512 image sizes correctly.

## Results

Measured on the held-out validation split, against a bicubic-upsampling baseline:

| Metric | Bicubic Baseline | Our Model |
|---|---|---|
| PSNR | 22.98 dB | 29.13 dB |
| SSIM | 0.5243 | 0.7923 |
| LPIPS | 0.4519 | 0.2521 |

**End-to-end runtime:** 33.4 ms/image (batch size 8), measured on an NVIDIA Tesla T4.
Includes disk read, preprocessing, CPU↔GPU transfer, model execution, and saving —
per the problem statement's runtime definition. Re-measured on the actual evaluation
hardware (H100) where required.

## Repository Structure

```
EPOCH-42/
├── run.py                  # standalone inference entry point
├── requirements.txt        # Python dependencies with version details
├── README.md               # setup, execution, and solution documentation
│
├── models/
│   ├── model.py            # model architecture definition
│   ├── best_model.pth      # trained checkpoint
│   └── config.json         # scale factor and target resolution config
│
└── result/
    ├── best_cases.png      # qualitative examples: best restorations
    └── worst_cases.png     # qualitative examples: failure cases
```

## Training

To reproduce `models/best_model.pth` from scratch:

```bash
python train.py --data_root /path/to/dataset --output_dir models/
```

- `--data_root` should contain, at any depth, a `NoisyLR/` folder and a `GT/` folder of
  paired `.npy` files with matching filenames.
- If `--data_root` is omitted, the script auto-detects these folders under
  `/kaggle/input` or `/kaggle/working`, so it also runs unmodified inside the original
  Kaggle notebook environment with no changes.
- Fixed seed (`42`) for Python, NumPy, and PyTorch; deterministic 90/10 train/val split.
- 50 epochs, batch size 8, AdamW (lr=1e-3, weight decay=1e-4), cosine annealing schedule,
  mixed-precision (AMP) training.
- Loss: Charbonnier + SSIM + gradient (Sobel) loss, combined as
  `Charbonnier + 1.0*SSIM + 0.1*Gradient`.
- Augmentation: random horizontal/vertical flips and 90°-multiple rotations (train split only).
- Model selection: best checkpoint chosen by `balanced_score = val_PSNR + 100*val_SSIM`
  on the held-out validation split (not used for training).
- Outputs: `best_model.pth`, `config.json` (records the detected scale factor and
  target resolution used by `run.py`), and `training_history.json` (per-epoch metrics).

`train.py` imports the model architecture from the same `models/model.py` that
`run.py` uses at inference time, so training and inference can never diverge on
architecture.

## External Resources

No external pretrained weights or external datasets were used beyond the official
KLA-provided GT/NoisyLR training pairs.

## Team

Epoch-42 — SEMICON India Hackathon 2026, KLA Problem Statement.
