"""
export_target.py — CLI + importable helper for converting UI JSON → TargetSpec.

CLI usage:
    conda run -n helios python ui/export_target.py --input zones.json --output target.pt

Notebook usage:
    import sys; sys.path.insert(0, '/Users/aryan/Projects/Helios')
    from ui.export_target import load_target_from_zones_json, load_target_from_pt

    target = load_target_from_zones_json('zones.json')
    # or load a previously exported spec:
    target = load_target_from_pt('target_spec.pt')
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from scripts.target_generation import load_target_from_zones_json
from scripts.targetSpec import TargetSpec

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_target_from_pt(pt_path: str | Path) -> TargetSpec:
    """Load a previously saved TargetSpec from a .pt file."""
    return torch.load(str(pt_path), weights_only=False)


def main():
    parser = argparse.ArgumentParser(description="Convert Helios UI zone JSON to TargetSpec .pt")
    parser.add_argument("--input", required=True, help="Path to zones JSON file")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--resolution", type=float, default=0.5, help="Grid resolution in degrees")
    args = parser.parse_args()

    print(f"Loading zones from {args.input}…")
    target = load_target_from_zones_json(args.input, args.resolution)
    out = Path(args.output)
    torch.save(target, out)
    print(f"TargetSpec saved → {out}")
    print(f"  Grid shape:  {target.targetShape}")
    print(
        f"  Power map:   min={target.powerMap.min():.3f}, max={target.powerMap.max():.3f}"
    )
    print(
        f"  Importance:  min={target.importanceMap.min():.3f}, max={target.importanceMap.max():.3f}"
    )


if __name__ == "__main__":
    main()
