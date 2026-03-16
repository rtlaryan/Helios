from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.arrayBatch import ArrayBatch

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def export_batch_for_ui(batch: ArrayBatch, json_path: str | Path, sample_id: int = 0):
    """
    Exports a single sample from an ArrayBatch to a JSON file that can be
    imported manually into the Helios UI (via the 'Import ArrayBatch JSON' button).
    """
    x = batch.elementLocalPosition[sample_id, 0, :].cpu().tolist()
    y = batch.elementLocalPosition[sample_id, 1, :].cpu().tolist()
    z = batch.elementLocalPosition[sample_id, 2, :].cpu().tolist()

    wr = batch.weights[sample_id, :].real.cpu().tolist()
    wi = batch.weights[sample_id, :].imag.cpu().tolist()

    lat = batch.LLAPosition[sample_id, 0].item()
    lon = batch.LLAPosition[sample_id, 1].item()
    alt = batch.LLAPosition[sample_id, 2].item()

    gain = batch.gain[sample_id]
    if gain.numel() > 1:
        gain = gain.cpu().tolist()
    else:
        gain = gain.item()

    data = {
        "positions_m": {"x": x, "y": y, "z": z},
        "weights_real": wr,
        "weights_imag": wi,
        "wavelength_m": float(batch.wavelength),
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "gain": gain,
    }

    out_path = Path(json_path)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Batch sample {sample_id} exported for UI visualization -> {out_path}")


def export_batch_from_pt_for_ui(pt_path: str | Path, json_path: str | Path, sample_id: int = 0):
    import torch

    batch = torch.load(str(pt_path), weights_only=False)
    export_batch_for_ui(batch, json_path, sample_id)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert ArrayBatch .pt to Helios UI JSON")
    parser.add_argument("--input", required=True, help="Path to input ArrayBatch .pt file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--sample", type=int, default=0, help="Batch sample index")
    args = parser.parse_args()

    print(f"Loading ArrayBatch from {args.input}…")
    export_batch_from_pt_for_ui(args.input, args.output, args.sample)


if __name__ == "__main__":
    main()
