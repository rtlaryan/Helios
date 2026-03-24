/**
 * api.js — Thin wrapper around the Flask backend.
 */
const API_BASE = 'http://localhost:5050';

window.HeliosAPI = {
    async _fetch(path, opts = {}) {
        const res = await fetch(API_BASE + path, {
            headers: { 'Content-Type': 'application/json' },
            ...opts,
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'Server error');
        return data;
    },

    // Ping the server
    async ping() {
        try {
            const res = await fetch(API_BASE + '/');
            return res.ok;
        } catch { return false; }
    },

    // Generate target maps from zone list
    async generateMaps(zones, resolutionDeg = 0.5, lat_range = null, lon_range = null, normalize = false) {
        return this._fetch('/api/target/generate', {
            method: 'POST',
            body: JSON.stringify({ zones, resolution_deg: resolutionDeg, lat_range, lon_range, normalize }),
        });
    },

    // Fetch full map data
    async getMaps() {
        return this._fetch('/api/target/data');
    },

    // Export TargetSpec
    async exportTarget(path, zones = []) {
        return this._fetch('/api/target/export', {
            method: 'POST',
            body: JSON.stringify({ path, zones }),
        });
    },

    // Import TargetSpec
    async importTarget(file_base64) {
        return this._fetch('/api/target/import', {
            method: 'POST',
            body: JSON.stringify({ file_base64 }),
        });
    },

    // Load antenna / generate batch
    async loadAntenna(params) {
        return this._fetch('/api/antenna', {
            method: 'POST',
            body: JSON.stringify(params),
        });
    },

    // Compute radiation pattern
    async computePattern(specParams, resolution = 150) {
        return this._fetch('/api/pattern', {
            method: 'POST',
            body: JSON.stringify({ spec_params: specParams, resolution }),
        });
    },

    // Ground projection
    async groundProjection(resolutionDeg = 2.0) {
        return this._fetch('/api/ground-projection', {
            method: 'POST',
            body: JSON.stringify({ resolution_deg: resolutionDeg }),
        });
    },

    // Import a manually built ArrayBatch from JSON object
    async importBatch(batchJson) {
        return this._fetch('/api/batch/import', {
            method: 'POST',
            body: JSON.stringify(batchJson),
        });
    },
};

// ── Server connectivity indicator ─────────────────────────────────────────────
async function checkServer() {
    const dot = document.getElementById('server-dot');
    const label = document.getElementById('server-label');
    dot.className = 'status-dot loading';
    label.textContent = 'Connecting…';
    const ok = await HeliosAPI.ping();
    if (ok) {
        dot.className = 'status-dot connected';
        label.textContent = 'Server ready';
    } else {
        dot.className = 'status-dot error';
        label.textContent = 'Server offline';
        toast('Flask server not reachable at :5050', 'error', 6000);
    }
}

// Check immediately and every 10s
checkServer();
setInterval(checkServer, 10000);
