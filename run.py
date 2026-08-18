"""
Standalone inference script.
Usage:
    python run.py <input_dir> <output_dir>

input_dir and output_dir are positional, per the required entry-point contract
(python run.py <input-dir> <output-dir>). --model_path/--config/--batch_size remain
available as optional flags for local debugging, but are never required - they default
to best_model.pth / config.json sitting next to this script.
--batch_size (default 8) batches images together when GPU memory permits, per the PDF's
"batch processing is preferred" guidance. Images are grouped by their (H, W) shape before
batching, since the hidden test set mixes ~256x256 and ~512x512 inputs (PDF Section 4A) and
a batch tensor requires uniform shape - files of a shape that don't fill a full batch are
still processed together in the remainder, just a smaller batch.
"""
import os
import sys
import glob
import json
import time
import argparse
from collections import defaultdict
import numpy as np
import torch

# model.py may sit either next to this script, or inside a models/ subfolder
# (the required submission layout is team_name/models/) - support both so the
# `from model import ...` below resolves either way.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_models_dir = os.path.join(_script_dir, "models")
if os.path.isdir(_models_dir):
    sys.path.insert(0, _models_dir)
else:
    sys.path.insert(0, _script_dir)

from model import HighResSemiconductorNet


def run_inference(input_dir, output_dir, model_path, config_path, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config_path) as f:
        cfg = json.load(f)
    scale_factor = cfg["scale_factor"]

    model = HighResSemiconductorNet(scale_factor=scale_factor).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"Running inference on {len(files)} files (batch_size={batch_size})...")

    # Group files by input shape first - required because torch.stack needs uniform shapes,
    # and the hidden test set mixes multiple resolutions per the PDF.
    shape_groups = defaultdict(list)
    for fname in files:
        # mmap avoids loading full pixel data just to check shape
        arr = np.load(fname, mmap_mode='r')
        shape_groups[arr.shape].append(fname)

    t0 = time.time()
    with torch.no_grad():
        for shape, group_files in shape_groups.items():
            for i in range(0, len(group_files), batch_size):
                batch_files = group_files[i:i + batch_size]
                batch_arrs = []
                for fname in batch_files:
                    arr = np.load(fname).astype(np.float32)
                    # NOTE: intentionally NOT dividing by 255 here even if max() > 1.0. This
                    # dataset's GT is normalized to [0,1], but NoisyLR values can legitimately
                    # overshoot slightly above 1.0 from noise - that does NOT mean this is a
                    # raw 0-255 image (confirmed explicitly in the PDF: "NoisyLR values may
                    # extend slightly outside [0,1]; this is intentional" and "KLA does not
                    # clip or renormalize outputs"). Training's dataset class only rescales
                    # when GT's max exceeds 1.0 (which it never does for this dataset), so
                    # noisy images were trained on as-is. Checking the noisy array's own max
                    # here (as an earlier version of this script did) creates a ~255x train/
                    # inference scale mismatch, starving the model of real signal and
                    # producing a near-black output image.
                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, axis=0)
                    batch_arrs.append(arr)

                inp = torch.from_numpy(np.stack(batch_arrs, axis=0)).to(device)

                # Target size is computed per-batch from this batch's own input shape x the
                # trained scale factor - NOT a fixed size from config. The hidden test set
                # mixes ~256x256 and ~512x512 images (PDF Section 4A), so a fixed target
                # would upscale every image to the same size regardless of its actual input
                # resolution. Safe here because every image in a batch shares the same shape
                # (batches are grouped by shape above).
                in_h, in_w = inp.shape[-2:]
                target_hw = (in_h * scale_factor, in_w * scale_factor)

                out1 = model(inp, target_hw=target_hw)
                out2 = torch.flip(model(torch.flip(inp, dims=[3]), target_hw=target_hw), dims=[3])
                out = torch.clamp((out1 + out2) / 2.0, 0.0, 1.0)
                out_np = out.cpu().numpy()
                # Safety net: guarantee no NaN/Inf ever reaches disk, even if a pathological
                # input triggers one somewhere upstream of the clamp (e.g. inside the model).
                out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)

                for j, fname in enumerate(batch_files):
                    restored = out_np[j, 0]  # drop channel dim -> (H, W) grayscale
                    np.save(os.path.join(output_dir, os.path.basename(fname)), restored)

    total = time.time() - t0
    print(f"Done in {total:.2f}s ({total/max(1,len(files)):.4f}s/image, "
          f"{len(shape_groups)} distinct input shape(s))")


if __name__ == "__main__":
    # Resolve default weight/config paths relative to this script's own location, not the
    # caller's current working directory - so the documented command works correctly
    # regardless of which directory the evaluator runs it from.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolve(filename):
        # Prefer models/<filename> (the required submission layout: team_name/models/),
        # fall back to the script's own directory for local/dev convenience.
        models_path = os.path.join(script_dir, "models", filename)
        if os.path.exists(models_path):
            return models_path
        return os.path.join(script_dir, filename)

    default_model_path = _resolve("best_model.pth")
    default_config_path = _resolve("config.json")

    p = argparse.ArgumentParser(
        description="Usage: python run.py <input_dir> <output_dir>")
    # Positional, per the required entry-point contract: python run.py <input-dir> <output-dir>
    p.add_argument("input_dir", type=str, help="Directory of degraded .npy files to restore.")
    p.add_argument("output_dir", type=str, help="Directory to write restored .npy files to.")
    # Optional overrides for local debugging only - never required for the evaluator's call.
    p.add_argument("--model_path", type=str, default=default_model_path)
    p.add_argument("--config", type=str, default=default_config_path)
    p.add_argument("--batch_size", type=int, default=8,
                    help="Images sharing the same input shape are batched together up to "
                         "this size. Lower this if you hit a CUDA out-of-memory error.")
    args = p.parse_args()
    run_inference(args.input_dir, args.output_dir, args.model_path, args.config, args.batch_size)
