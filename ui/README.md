# Helios UI

Interactive beamforming design tool built on a **Flask** backend + **vanilla JS/Three.js** frontend.

## Quick Start

Run from the project root (conda env must be active):

```bash
# Option A — module invocation (recommended)
conda run -n helios python -m ui.server

# Option B — activate first
conda activate helios
python -m ui.server
```

Open **http://localhost:5050** in a browser.

---

## Architecture

```
ui/
├── server.py          # Flask REST API — routes and global state
├── helios_bridge.py   # Pure conversion layer: JSON ↔ Helios dataclasses
├── index.html         # Single-page application shell
├── css/
│   └── main.css       # All styling (dark theme, design tokens)
└── js/
    ├── state.js        # Shared HeliosState singleton + event bus + UI helpers
    ├── api.js          # HeliosAPI — thin fetch wrapper over the Flask backend
    ├── globe.js        # GlobeRenderer — Three.js Earth globe
    ├── targetEditor.js # TargetEditor — zone drawing on the globe (circle / polygon)
    ├── mapRenderer.js  # MapRenderer — power/importance map generation & preview
    ├── antennaPanel.js # AntennaPanel — antenna configuration & loading
    ├── patternPanel.js # PatternPanel — az/el charts + 3D radiation lobe
    ├── arrayGeoPanel.js# ArrayGeoPanel — standalone Array Geometry header-tab view
    └── app.js          # Entry point — initialises all modules on DOMContentLoaded
```

Script load order matters (globals are injected on `window`):

```
state.js → api.js → globe.js → targetEditor.js → mapRenderer.js
         → antennaPanel.js → patternPanel.js → arrayGeoPanel.js → app.js
```

---

## REST API

All endpoints return `{ "ok": true, ...payload }` on success or `{ "ok": false, "error": "..." }` on failure.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/target/data` | Fetch the current power/importance map arrays |
| `POST` | `/api/target/generate` | Generate maps from zone list `{ zones, resolution_deg, lat_range?, lon_range? }` |
| `POST` | `/api/target/import` | Import a `TargetSpec .pt` file `{ file_base64 }` |
| `POST` | `/api/target/export` | Save current maps as a `TargetSpec .pt` file `{ path?, zones }` |
| `POST` | `/api/antenna` | Build antenna from spec params; returns element positions + weights |
| `GET`  | `/api/array-geometry` | Return cached antenna geometry |
| `POST` | `/api/batch/import` | Import an `ArrayBatch` from a JSON payload |
| `POST` | `/api/pattern` | Compute az/el cuts + 3D radiation pattern `{ spec_params?, resolution }` |
| `POST` | `/api/ground-projection` | Project beam pattern onto Earth surface `{ resolution_deg }` |

---

## Module Notes

### `server.py`
Single-user, in-memory state (`_current_batch`, `_current_maps`). Designed as a local desktop tool — no authentication or multi-session support.

### `helios_bridge.py`
All heavy computation (coordinate transforms, array simulation, map rasterisation) lives here. `server.py` stays thin — it only handles HTTP concerns.

### `globe.js`
Loads NASA Blue Marble textures from a CDN; falls back to a procedural canvas texture if the network is unavailable. Zone meshes (circles and polygons) are attached as children of `earthMesh` so they rotate with the Earth.

### `state.js`
Acts as a lightweight event bus (`HeliosState.on` / `HeliosState.emit`). Modules communicate through events rather than direct cross-module calls, keeping coupling low.

### `patternPanel.js`
The `addAxisLines` helper is shared between the 3D pattern viewer and the array-geometry sub-panel (both show the same XYZ triad).
