"""
server.py  –  Helios UI Flask backend.

Run from the project root:
    conda run -n helios python -m ui.server

Or:
    cd /Users/aryan/Projects/Helios
    conda activate helios
    python -m ui.server
"""

from __future__ import annotations

import base64
import io
import sys
import traceback
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from scripts.arrayBatch import ArrayBatch
from scripts.batchFactory import generateBatch
from scripts.coordinateTransforms import LLAtoECEF
from scripts.targetSpec import TargetSpec

from ui.helios_bridge import (
    batch_to_json,
    build_array_spec,
    build_target_maps,
    build_target_spec,
    compute_ground_projection,
    compute_pattern_2d,
)

# ── Ensure project root is on the path ───────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
CORS(app)

DEVICE = torch.device("cpu")
DTYPE = torch.float32

# In-memory state (single user, local tool)
_current_batch: "ArrayBatch | None" = None  # noqa: F821
_current_maps: dict | None = None

UI_DIR = Path(__file__).parent


# ── Static file serving ───────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory(str(UI_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(UI_DIR), filename)


# ── Target map API ────────────────────────────────────────────────────────────


@app.route("/api/target/generate", methods=["POST"])
def target_generate():
    """
    Body: { zones: [...], resolution_deg: float, lat_range: [min,max]?, lon_range: [min,max]? }
    """
    global _current_maps
    try:
        body = request.get_json(force=True)
        zones = body.get("zones", [])
        resolution = float(body.get("resolution_deg", 0.5))
        lat_range = body.get("lat_range")  # [min, max] or None
        lon_range = body.get("lon_range")  # [min, max] or None
        normalize = bool(body.get("normalize", False))
        maps = build_target_maps(
            zones,
            resolution_deg=resolution,
            lat_range=tuple(lat_range) if lat_range else None,
            lon_range=tuple(lon_range) if lon_range else None,
            normalize=normalize,
        )
        _current_maps = maps
        return jsonify(
            {
                "ok": True,
                "shape": maps["shape"],
                "lat_range": [maps["lat_vec"][0], maps["lat_vec"][-1]],
                "lon_range": [maps["lon_vec"][0], maps["lon_vec"][-1]],
            }
        )
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


@app.route("/api/target/data", methods=["GET"])
def target_data():
    """Return the full map arrays (may be large)."""
    if _current_maps is None:
        return jsonify({"ok": False, "error": "No maps generated yet"}), 404
    return jsonify({"ok": True, **_current_maps})


@app.route("/api/target/import", methods=["POST"])
def target_import():
    """
    Import an existing TargetSpec .pt file.
    Body: { "file_base64": "<base64-encoded .pt file>" }
    """
    global _current_maps
    try:
        body = request.get_json(force=True) or {}
        b64 = body.get("file_base64")
        if not b64:
            return jsonify({"ok": False, "error": "No 'file_base64' payload provided"}), 400

        pt_data = base64.b64decode(b64)
        target = torch.load(io.BytesIO(pt_data), weights_only=False)
        if not isinstance(target, TargetSpec):
            return jsonify(
                {"ok": False, "error": "File does not contain a TargetSpec instance"}
            ), 400

        # Extract mapping data to push to frontend.
        # lat_grid/lon_grid are meshgrids (indexing="xy") and monotonically increasing.
        lat_grid = target.searchLatitudes.cpu().numpy()
        lon_grid = target.searchLongitudes.cpu().numpy()
        pwr_map = target.powerMap.cpu().numpy()
        imp_map = target.importanceMap.cpu().numpy()

        lat_vec = lat_grid[:, 0].tolist()
        lon_vec = lon_grid[0, :].tolist()

        maps = {
            "shape": list(lat_grid.shape),
            "lat_vec": lat_vec,
            "lon_vec": lon_vec,
            "power_map": pwr_map.tolist(),
            "importance_map": imp_map.tolist(),
        }

        _current_maps = maps
        return jsonify({"ok": True, **maps})

    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


@app.route("/api/target/export", methods=["POST"])
def target_export():
    """Save a TargetSpec to disk as a .pt file."""
    if _current_maps is None:
        return jsonify({"ok": False, "error": "No maps to export"}), 400
    try:
        body = request.get_json(force=True) or {}
        zones = body.get("zones", [])
        out_path = Path(body.get("path", str(_ROOT / "target_spec.pt")))
        spec = build_target_spec(_current_maps, zones)
        torch.save(spec, out_path)
        return jsonify({"ok": True, "saved_to": str(out_path)})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


# ── Antenna / Array API ───────────────────────────────────────────────────────


@app.route("/api/antenna", methods=["POST"])
def antenna():
    """
    Body: antenna spec params (see build_array_spec).
    Returns element positions + weights for sample 0.
    """
    global _current_batch
    try:
        params = request.get_json(force=True) or {}
        spec = build_array_spec(params)
        batch = generateBatch(spec, batchSize=1, device=DEVICE, dtype=DTYPE, weightsType="uniform")
        _current_batch = batch
        data = batch_to_json(batch, sample_id=0)
        return jsonify({"ok": True, **data})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


@app.route("/api/array-geometry", methods=["GET"])
def array_geometry():
    if _current_batch is None:
        return jsonify({"ok": False, "error": "No antenna loaded yet"}), 404
    return jsonify({"ok": True, **batch_to_json(_current_batch, sample_id=0)})


@app.route("/api/batch/import", methods=["POST"])
def batch_import():
    """
    Import a manually built ArrayBatch from JSON.

    Body:
    {
      "positions_m":  {"x": [...], "y": [...], "z": [...]},  // element positions in metres
      "weights_real": [...],        // real parts of complex weights
      "weights_imag": [...],        // imaginary parts (default: all zeros)
      "wavelength_m": float,        // metres
      "lat":  float,                // degrees
      "lon":  float,                // degrees
      "alt":  float,                // metres
      "gain": float | [...]         // scalar or per-element (default: 1.0)
    }
    """
    global _current_batch
    try:
        body = request.get_json(force=True) or {}
        pos = body["positions_m"]
        x = torch.tensor(pos["x"], dtype=DTYPE)
        y = torch.tensor(pos["y"], dtype=DTYPE)
        z = torch.tensor(pos["z"], dtype=DTYPE)
        N = x.shape[0]

        # Element positions: [1, 3, N]
        elem_pos = torch.stack([x, y, z], dim=0).unsqueeze(0)

        # Complex weights: [1, N]
        wr = torch.tensor(body["weights_real"], dtype=DTYPE)
        wi = torch.tensor(body.get("weights_imag", [0.0] * N), dtype=DTYPE)
        weights = torch.complex(wr, wi).unsqueeze(0)

        wavelength = float(body["wavelength_m"])

        lat = float(body.get("lat", 0.0))
        lon = float(body.get("lon", 0.0))
        alt = float(body.get("alt", 3.6e7))

        lla = torch.tensor([[lat, lon, alt]], dtype=DTYPE)
        ecef = LLAtoECEF(lla)

        gain_raw = body.get("gain", 1.0)
        if isinstance(gain_raw, (int, float)):
            gain = torch.full((1,), float(gain_raw), dtype=DTYPE)
        else:
            gain = torch.tensor(gain_raw, dtype=DTYPE).unsqueeze(0)

        _current_batch = ArrayBatch(
            elementLocalPosition=elem_pos,
            weights=weights,
            wavelength=wavelength,
            gain=gain,
            LLAPosition=lla,
            ECEFPosition=ecef,
            elementMask=None,
        )

        data = batch_to_json(_current_batch, sample_id=0)
        return jsonify({"ok": True, **data})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


# ── Radiation pattern API ─────────────────────────────────────────────────────


@app.route("/api/pattern", methods=["POST"])
def pattern():
    """
    Body: { spec_params: {...}, resolution: int }
    Returns az/el cuts + 3D pattern arrays.
    """
    global _current_batch
    try:
        body = request.get_json(force=True) or {}
        resolution = int(body.get("resolution", 150))

        # If spec_params provided and non-null, regenerate batch; else reuse cached batch.
        if body.get("spec_params"):
            spec = build_array_spec(body["spec_params"])
            _current_batch = generateBatch(
                spec, batchSize=1, device=DEVICE, dtype=DTYPE, randomWeights=False
            )

        if _current_batch is None:
            return jsonify({"ok": False, "error": "No antenna loaded"}), 400

        data = compute_pattern_2d(_current_batch, sample_id=0, resolution=resolution)
        return jsonify({"ok": True, **data})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


@app.route("/api/ground-projection", methods=["POST"])
def ground_projection():
    """
    Project the beam pattern onto the Earth surface.
    Body: { resolution_deg: float }
    """
    if _current_batch is None:
        return jsonify({"ok": False, "error": "No antenna loaded"}), 400
    try:
        body = request.get_json(force=True) or {}
        res = float(body.get("resolution_deg", 2.0))
        proj = compute_ground_projection(_current_batch, 0, _current_maps or {}, res)
        return jsonify({"ok": True, **proj})
    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Helios UI  —  http://localhost:5050")
    print("  Ctrl-C to stop")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False)
