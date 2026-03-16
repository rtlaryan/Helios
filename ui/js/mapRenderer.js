/**
 * mapRenderer.js — Generates and displays power/importance map previews.
 */

const MapRenderer = (() => {

    // ── Canvas mini-map rendering ──────────────────────────────────────────────
    function drawPowerPreview(maps) {
        const canvas = document.getElementById('power-preview-canvas');
        if (!canvas || !maps) return;
        const H = maps.power_map.length, W = maps.power_map[0].length;
        canvas.width = W; canvas.height = H;
        canvas.style.height = '110px';
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(W, H);

        // Find global min/max
        let mn = Infinity, mx = -Infinity;
        maps.power_map.forEach(row => row.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; }));
        document.getElementById('power-min-label').textContent = mn.toFixed(0);
        document.getElementById('power-max-label').textContent = mx.toFixed(0) + ' dB';

        const range = mx - mn || 1;
        for (let r = 0; r < H; r++) {
            for (let c = 0; c < W; c++) {
                // lat_vec runs south→north, so row 0 of canvas = north: read from H-1-r
                const val = maps.power_map[H - 1 - r][c];
                const t   = (val - mn) / range;
                // Jet-like colormap
                let rv, gv, bv;
                if      (t < 0.25) { rv = 0;   gv = Math.round(t * 4 * 255);             bv = 255; }
                else if (t < 0.5)  { rv = 0;   gv = 255; bv = Math.round((1 - (t - 0.25) * 4) * 255); }
                else if (t < 0.75) { rv = Math.round((t - 0.5) * 4 * 255); gv = 255;     bv = 0; }
                else               { rv = 255; gv = Math.round((1 - (t - 0.75) * 4) * 255); bv = 0; }
                const idx = (r * W + c) * 4;
                img.data[idx] = rv; img.data[idx + 1] = gv; img.data[idx + 2] = bv;
                img.data[idx + 3] = val < mn + range * 0.05 ? 30 : 220;
            }
        }
        ctx.putImageData(img, 0, 0);
        drawGridOverlay(ctx, W, H, maps.lat_vec, maps.lon_vec);
    }

    function drawImportancePreview(maps) {
        const canvas = document.getElementById('importance-preview-canvas');
        if (!canvas || !maps) return;
        const H = maps.importance_map.length, W = maps.importance_map[0].length;
        canvas.width = W; canvas.height = H;
        canvas.style.height = '110px';
        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(W, H);
        for (let r = 0; r < H; r++) {
            for (let c = 0; c < W; c++) {
                const v   = maps.importance_map[H - 1 - r][c]; // north-up flip
                const idx = (r * W + c) * 4;
                img.data[idx]     = Math.round(v * 167);
                img.data[idx + 1] = Math.round(v * 139);
                img.data[idx + 2] = Math.round(50 + v * 205);
                img.data[idx + 3] = v < 0.02 ? 20 : Math.round(v * 230 + 25);
            }
        }
        ctx.putImageData(img, 0, 0);
        drawGridOverlay(ctx, W, H, maps.lat_vec, maps.lon_vec);
    }

    function drawGridOverlay(ctx, W, H, latVec, lonVec) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth   = 1;
        ctx.beginPath();

        const extentLon = lonVec[lonVec.length - 1] - lonVec[0];
        const step      = extentLon > 100 ? 30 : (extentLon > 40 ? 10 : 5);

        // Vertical lines (Longitude)
        for (let i = 0; i < W; i++) {
            if (Math.abs(lonVec[i] % step) < (lonVec[1] - lonVec[0])) {
                ctx.moveTo(i, 0); ctx.lineTo(i, H);
            }
        }
        // Horizontal lines (Latitude)
        for (let i = 0; i < H; i++) {
            // latVec is ascending; canvas row i maps to latVec[H-1-i]
            if (Math.abs(latVec[H - 1 - i] % step) < (latVec[1] - latVec[0])) {
                ctx.moveTo(0, i); ctx.lineTo(W, i);
            }
        }
        ctx.stroke();
    }

    // ── Statistics ─────────────────────────────────────────────────────────────
    function updateStats(maps) {
        if (!maps) return;
        let peak = -Infinity;
        let covered = 0, total = 0;
        maps.power_map.forEach(row => row.forEach(v => {
            if (v > peak) peak = v;
            if (v > -50) covered++;
            total++;
        }));
        document.getElementById('stat-peak-power').textContent = peak.toFixed(1) + ' dB';
        document.getElementById('stat-coverage').textContent   = ((covered / total) * 100).toFixed(1) + '%';
        document.getElementById('stat-grid-size').textContent  = `${maps.shape[0]}×${maps.shape[1]}`;
    }

    // ── Generate maps from current zones ──────────────────────────────────────
    async function generateMaps() {
        const zones = HeliosState.zones.map(z => {
            const base = {
                type:    z.type,
                shape:   z.shape || 'circle',
                peak_db: z.peak_db,
                rolloff: z.rolloff,
            };
            if (z.shape === 'polygon') {
                base.verts = z.verts;
            } else {
                base.lat        = z.lat;
                base.lon        = z.lon;
                base.radius_deg = z.radius_deg;
            }
            return base;
        });

        if (zones.length === 0) {
            toast('Draw at least one zone first', 'error');
            return;
        }

        const resolution  = parseFloat(document.getElementById('map-resolution')?.value || 0.5);
        const focusEnabled = document.getElementById('focus-enabled')?.checked;
        let lat_range = null, lon_range = null;
        if (focusEnabled) {
            lat_range = [
                parseFloat(document.getElementById('focus-lat-min').value),
                parseFloat(document.getElementById('focus-lat-max').value),
            ];
            lon_range = [
                parseFloat(document.getElementById('focus-lon-min').value),
                parseFloat(document.getElementById('focus-lon-max').value),
            ];
        }

        const btns = document.querySelectorAll('.btn-generate');
        btns.forEach(btn => { btn.disabled = true; btn.innerHTML = '<div class="spinner"></div> Generating…'; });

        try {
            await HeliosAPI.generateMaps(zones, resolution, lat_range, lon_range);
            const fullData = await HeliosAPI.getMaps();
            HeliosState.maps = fullData;

            drawPowerPreview(fullData);
            drawImportancePreview(fullData);
            updateStats(fullData);

            const overlayMode    = document.getElementById('map-display-mode')?.value || 'power';
            const overlayOpacity = parseFloat(document.getElementById('overlay-opacity')?.value || 0.75);
            GlobeRenderer.updateMapOverlay(fullData, overlayMode, overlayOpacity);

            toast('Maps generated successfully', 'success');
            HeliosState.emit('mapsGenerated', fullData);
        } catch (err) {
            toast('Map generation failed: ' + err.message, 'error');
            console.error(err);
        } finally {
            document.querySelectorAll('.btn-generate').forEach(btn => {
                btn.disabled = false;
                btn.innerHTML = `
                    <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="22,12 18,12 15,21 9,3 6,12 2,12"/></svg>
                    Generate Maps`;
            });
        }
    }

    // ── Export ─────────────────────────────────────────────────────────────────
    async function exportTarget() {
        if (!HeliosState.maps) {
            toast('Generate maps first', 'error');
            return;
        }
        try {
            const payloadZones = HeliosState.zones.map(z => z.shape === 'polygon' ? { ...z, verts: z.verts } : z);
            const result = await HeliosAPI.exportTarget(undefined, payloadZones);
            toast('Exported → ' + result.saved_to, 'success', 5000);
        } catch (err) {
            toast('Export failed: ' + err.message, 'error');
        }
    }

    function init() {
        document.querySelectorAll('.btn-generate').forEach(b => b.addEventListener('click', generateMaps));
        document.querySelectorAll('.btn-export').forEach(b => b.addEventListener('click', exportTarget));

        // Manual TargetSpec import via file picker
        document.getElementById('btn-import-target')?.addEventListener('click', () => {
            const fileInput = document.getElementById('target-import-file');
            if (!fileInput.files.length) { toast('Choose a .pt file first', 'error'); return; }
            const btn = document.getElementById('btn-import-target');
            btn.disabled = true; btn.textContent = '…';

            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    // e.target.result is "data:application/octet-stream;base64,AAAA..."
                    const base64data = e.target.result.split(',')[1];
                    const maps = await HeliosAPI.importTarget(base64data);
                    HeliosState.maps = maps;

                    drawPowerPreview(maps);
                    drawImportancePreview(maps);
                    updateStats(maps);

                    const mode    = document.getElementById('map-display-mode')?.value || 'power';
                    const opacity = parseFloat(document.getElementById('overlay-opacity')?.value || 0.75);
                    GlobeRenderer.updateMapOverlay(maps, mode, opacity);

                    toast(`Imported TargetSpec (${maps.shape[0]}×${maps.shape[1]})`, 'success');
                    HeliosState.emit('mapsGenerated', maps);
                } catch (err) {
                    toast('Import failed: ' + err.message, 'error');
                    console.error(err);
                } finally {
                    btn.disabled = false; btn.textContent = 'Import';
                }
            };
            reader.readAsDataURL(fileInput.files[0]);
        });

        // Live overlay controls
        const syncOverlay = () => {
            if (!HeliosState.maps) return;
            const mode    = document.getElementById('map-display-mode').value;
            const opacity = parseFloat(document.getElementById('overlay-opacity').value);
            GlobeRenderer.updateMapOverlay(HeliosState.maps, mode, opacity);
        };
        document.getElementById('map-display-mode')?.addEventListener('change', syncOverlay);
        document.getElementById('overlay-opacity')?.addEventListener('input',   syncOverlay);
    }

    return { init, drawPowerPreview, drawImportancePreview };
})();
