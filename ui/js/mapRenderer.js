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
        const isLinear = maps.power_normalized;

        // Find signal min/max — exclude the -100 dB floor sentinel (no-signal pixels)
        let mn = Infinity, mx = -Infinity;
        maps.power_map.forEach(row => row.forEach(v => {
            const skip = !isLinear && v <= -99.9;
            if (!skip) { if (v < mn) mn = v; if (v > mx) mx = v; }
        }));
        if (!isFinite(mn)) mn = isLinear ? 0 : -100;

        if (isLinear) {
            document.getElementById('power-min-label').textContent = '0.0';
            document.getElementById('power-max-label').textContent = '1.0 (linear)';
        } else {
            document.getElementById('power-min-label').textContent = mn.toFixed(0);
            document.getElementById('power-max-label').textContent = mx.toFixed(0) + ' dB';
        }

        const range = mx - mn || 1;
        for (let r = 0; r < H; r++) {
            for (let c = 0; c < W; c++) {
                const val = maps.power_map[H - 1 - r][c];
                const t   = Math.max(0, Math.min(1, (val - mn) / range));
                let rv, gv, bv, av;
                if (isLinear) {
                    // Purple→blue→white gradient matching importance style but distinct
                    rv = Math.round(t * 59  + (1 - t) * 4);
                    gv = Math.round(t * 130 + (1 - t) * 20);
                    bv = Math.round(t * 246 + (1 - t) * 80);
                    av = val < 0.01 ? 20 : Math.round(t * 200 + 30);
                } else {
                    // Jet-like colormap for dB
                    if      (t < 0.25) { rv = 0;   gv = Math.round(t * 4 * 255);             bv = 255; }
                    else if (t < 0.5)  { rv = 0;   gv = 255; bv = Math.round((1 - (t - 0.25) * 4) * 255); }
                    else if (t < 0.75) { rv = Math.round((t - 0.5) * 4 * 255); gv = 255;     bv = 0; }
                    else               { rv = 255; gv = Math.round((1 - (t - 0.75) * 4) * 255); bv = 0; }
                    av = val < mn + range * 0.05 ? 30 : 220;
                }
                const idx = (r * W + c) * 4;
                img.data[idx] = rv; img.data[idx + 1] = gv; img.data[idx + 2] = bv;
                img.data[idx + 3] = av;
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
            const buf = parseFloat(document.getElementById('focus-buffer')?.value ?? 10);
            const allZones = HeliosState.zones;
            let latMin = 90, latMax = -90, lonMin = 180, lonMax = -180;
            for (const z of allZones) {
                if (z.shape === 'polygon') {
                    for (const v of z.verts) {
                        latMin = Math.min(latMin, v.lat); latMax = Math.max(latMax, v.lat);
                        lonMin = Math.min(lonMin, v.lon); lonMax = Math.max(lonMax, v.lon);
                    }
                } else {
                    const r = z.radius_deg || 8;
                    latMin = Math.min(latMin, z.lat - r); latMax = Math.max(latMax, z.lat + r);
                    lonMin = Math.min(lonMin, z.lon - r); lonMax = Math.max(lonMax, z.lon + r);
                }
            }
            lat_range = [Math.max(-90,  latMin - buf), Math.min(90,  latMax + buf)];
            lon_range = [Math.max(-180, lonMin - buf), Math.min(180, lonMax + buf)];
            const preview = document.getElementById('focus-bounds-preview');
            if (preview) preview.textContent =
                `lat [${lat_range[0].toFixed(0)}°, ${lat_range[1].toFixed(0)}°]  ` +
                `lon [${lon_range[0].toFixed(0)}°, ${lon_range[1].toFixed(0)}°]`;
        } else {
            const preview = document.getElementById('focus-bounds-preview');
            if (preview) preview.textContent = '';
        }

        const btns = document.querySelectorAll('.btn-generate');
        btns.forEach(btn => { btn.disabled = true; btn.innerHTML = '<div class="spinner"></div> Generating…'; });

        const normalize = document.getElementById('zone-normalize')?.checked ?? false;

        try {
            await HeliosAPI.generateMaps(zones, resolution, lat_range, lon_range, normalize);
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
