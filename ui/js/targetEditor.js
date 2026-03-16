/**
 * targetEditor.js — Handles drawing zones on the globe.
 * Modes: 'circle' | 'polygon'
 */

const TargetEditor = (() => {
    const COLORS_POWER      = [0xfbbf24, 0xf97316, 0xef4444, 0xfde68a, 0xfca5a5];
    const COLORS_IMPORTANCE = [0xa78bfa, 0x818cf8, 0x38bdf8, 0xc084fc, 0x6ee7b7];
    let _colorIdxPower = 0, _colorIdxImp = 0;

    let _drawMode = 'circle';
    let _canvas;

    // ── Polygon state ──────────────────────────────────────────────────────────
    let _polyVerts = [];       // [{lat, lon}]
    let _polyPreviewLine = null;

    function nextColor(type) {
        return type === 'power'
            ? COLORS_POWER[(_colorIdxPower++) % COLORS_POWER.length]
            : COLORS_IMPORTANCE[(_colorIdxImp++) % COLORS_IMPORTANCE.length];
    }

    // ── Polygon helpers ────────────────────────────────────────────────────────

    /** Compute bounding circle: lat/lon centroid + max great-circle radius (degrees). */
    function polygonBoundingCircle(verts) {
        const n    = verts.length;
        const clat = verts.reduce((s, v) => s + v.lat, 0) / n;
        const clon = verts.reduce((s, v) => s + v.lon, 0) / n;
        const toRad = d => d * Math.PI / 180;
        function haversine(la1, lo1, la2, lo2) {
            const dLat = toRad(la2 - la1), dLon = toRad(lo2 - lo1);
            const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(la1)) * Math.cos(toRad(la2)) * Math.sin(dLon / 2) ** 2;
            return 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)) * 180 / Math.PI;
        }
        const radius_deg = Math.max(...verts.map(v => haversine(clat, clon, v.lat, v.lon)));
        return { lat: clat, lon: clon, radius_deg: Math.max(radius_deg, 0.5) };
    }

    function clearPolyPreview() {
        if (_polyPreviewLine) {
            GlobeRenderer.removePolyPreview(_polyPreviewLine);
            _polyPreviewLine = null;
        }
        _polyVerts = [];
    }

    function updatePolyPreview() {
        if (_polyPreviewLine) GlobeRenderer.removePolyPreview(_polyPreviewLine);
        if (_polyVerts.length < 2) { _polyPreviewLine = null; return; }
        _polyPreviewLine = GlobeRenderer.addPolyPreviewLine(_polyVerts);
    }

    function finalisePolygon() {
        if (_polyVerts.length < 3) {
            toast('Need at least 3 vertices for a polygon zone', 'error');
            clearPolyPreview();
            return;
        }

        const type   = HeliosState.zoneType;
        const params = HeliosState.zoneParams;
        const color  = nextColor(type);

        const id = GlobeRenderer.addPolygonZoneMesh([..._polyVerts], color, 0.30);
        HeliosState.zones.push({
            id, type,
            shape: 'polygon',
            verts: [..._polyVerts],
            peak_db: params.peak_db,
            rolloff: params.rolloff,
            color,
            meshId: id,
        });

        clearPolyPreview();
        HeliosState.emit('zonesChanged');
        toast(`Polygon zone — ${_polyVerts.length} vertices`, 'success');
    }

    // ── Click handlers ─────────────────────────────────────────────────────────
    function handleClick(e) {
        if (e.button !== 0) return;
        if (!e.shiftKey) return; // must hold Shift to draw

        const hit = GlobeRenderer.rayCastLatLon(e);
        if (!hit) return;

        if (_drawMode === 'circle') {
            const params = HeliosState.zoneParams;
            const type   = HeliosState.zoneType;
            const color  = nextColor(type);
            const zoneId = GlobeRenderer.addZoneMesh(hit.lat, hit.lon, params.radius_deg, color, 0.35);
            HeliosState.zones.push({
                id: zoneId, type,
                lat: hit.lat, lon: hit.lon,
                radius_deg: params.radius_deg,
                peak_db: params.peak_db,
                rolloff:  params.rolloff,
                color, meshId: zoneId,
            });
            HeliosState.emit('zonesChanged');
            e.preventDefault();
        }

        if (_drawMode === 'polygon') {
            _polyVerts.push({ lat: hit.lat, lon: hit.lon });
            updatePolyPreview();
            e.preventDefault();
        }
    }

    // Right-click in polygon mode → close polygon
    function handleRightClick(e) {
        if (_drawMode !== 'polygon') return;
        e.preventDefault();
        finalisePolygon();
    }

    // Double-click in polygon mode → also close
    function handleDblClick(e) {
        if (_drawMode !== 'polygon') return;
        e.preventDefault();
        finalisePolygon();
    }

    // ── Zone list UI ───────────────────────────────────────────────────────────
    function updateZoneListUI() {
        const body    = document.getElementById('zone-list-body');
        const hint    = document.getElementById('zone-empty-hint');
        const badge   = document.getElementById('zone-count-badge');
        const overlay = document.getElementById('zone-list-overlay');

        badge.textContent = `${HeliosState.zones.length} zones`;

        // Clear existing rows
        Array.from(body.children).forEach(c => { if (c !== hint) c.remove(); });
        overlay.innerHTML = '';

        if (HeliosState.zones.length === 0) {
            hint.style.display = '';
            return;
        }
        hint.style.display = 'none';

        HeliosState.zones.forEach((z, i) => {
            const colorHex = '#' + z.color.toString(16).padStart(6, '0');
            const isImp    = z.type === 'importance';
            const isPoly   = z.shape === 'polygon';
            const metaStr  = isImp ? `σ=${z.rolloff}°` : `${z.peak_db}dB · σ=${z.rolloff}°`;
            const shapeStr = isPoly ? ` · ${z.verts.length}pts` : ` r=${z.radius_deg?.toFixed(1)}°`;

            // Sidebar row
            const row = document.createElement('div');
            row.style.cssText = `
                display:flex; align-items:center; gap:8px; padding:6px 8px;
                background:var(--bg-input); border-radius:6px; margin-bottom:6px;
            `;
            row.innerHTML = `
                <span style="width:10px;height:10px;border-radius:${isPoly ? '2px' : '50%'};background:${colorHex};flex-shrink:0;"></span>
                <span style="flex:1;font-size:12px;color:var(--text-primary);font-weight:500;">
                    ${isImp ? 'Imp' : 'Pwr'} ${isPoly ? 'Poly' : 'Zone'} ${i + 1}
                </span>
                <span style="font-size:10.5px;font-family:var(--font-mono);color:var(--text-dim);">
                    ${metaStr}${shapeStr}
                </span>
                <button data-id="${z.id}" style="
                    width:18px;height:18px;border:none;background:none;cursor:pointer;
                    color:var(--text-dim);font-size:14px;display:flex;align-items:center;justify-content:center;
                    border-radius:50%;
                " title="Remove zone">×</button>
            `;
            row.querySelector('button').addEventListener('click', () => deleteZone(z.id));
            body.appendChild(row);

            // Globe overlay chip — only show lat/lon for circle zones
            const chip = document.createElement('div');
            chip.className = 'zone-chip';
            chip.innerHTML = `
                <span class="zone-color" style="background:${colorHex}"></span>
                <span class="zone-label">${isImp ? 'Importance' : 'Power'} ${i + 1}</span>
                ${!isPoly ? `<span class="zone-meta">${z.lat.toFixed(1)}°, ${z.lon.toFixed(1)}°</span>` : ''}
            `;
            overlay.appendChild(chip);
        });

        document.getElementById('stat-zones').textContent = HeliosState.zones.length;
    }

    function deleteZone(id) {
        GlobeRenderer.removeZoneMesh(id);
        HeliosState.zones = HeliosState.zones.filter(z => z.id !== id);
        HeliosState.emit('zonesChanged');
    }

    // ── Draw mode buttons ──────────────────────────────────────────────────────
    function setDrawMode(mode) {
        if (_drawMode === 'polygon' && mode !== 'polygon') clearPolyPreview();

        _drawMode = mode;
        HeliosState.drawMode = mode;
        GlobeRenderer.setOrbitEnabled(true);

        document.querySelectorAll('.draw-btn').forEach(b => b.classList.remove('active'));
        const btnMap = { circle: 'btn-circle', polygon: 'btn-polygon' };
        document.getElementById(btnMap[mode])?.classList.add('active');

        _canvas.style.cursor = 'crosshair';

        const hint = document.getElementById('poly-hint');
        if (hint) hint.style.display = mode === 'polygon' ? '' : 'none';
    }

    function init(canvasId) {
        _canvas = document.getElementById(canvasId);

        document.getElementById('btn-circle')?.addEventListener('click',  () => setDrawMode('circle'));
        document.getElementById('btn-polygon')?.addEventListener('click', () => setDrawMode('polygon'));

        document.getElementById('btn-clear-zones')?.addEventListener('click', () => {
            HeliosState.zones.forEach(z => GlobeRenderer.removeZoneMesh(z.id));
            HeliosState.zones = [];
            clearPolyPreview();
            HeliosState.emit('zonesChanged');

            // Clear globe overlays
            GlobeRenderer.updateGroundProjection(null);
            GlobeRenderer.updateMapOverlay(null, 'none', 0);
            document.getElementById('map-display-mode').value = 'none';
        });

        _canvas.addEventListener('click',       handleClick);
        _canvas.addEventListener('contextmenu', handleRightClick);
        _canvas.addEventListener('dblclick',    handleDblClick);

        // Shift+scroll → resize zone radius
        _canvas.addEventListener('wheel', e => {
            e.preventDefault();
            if (e.shiftKey && _drawMode === 'circle') {
                const slider = document.getElementById('zone-radius');
                let rv = parseFloat(slider.value) || 8;
                rv += e.deltaY < 0 ? 0.5 : -0.5;
                rv = Math.max(1, Math.min(60, rv));
                slider.value = rv;
                slider.dispatchEvent(new Event('input'));
            }
        });

        // Keyboard shortcuts: P = power, I = importance
        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const key = e.key.toLowerCase();
            if (key === 'p') document.querySelector('.zone-type-btn[data-type="power"]')?.click();
            if (key === 'i') document.querySelector('.zone-type-btn[data-type="importance"]')?.click();
        });

        HeliosState.on('zonesChanged', updateZoneListUI);
    }

    return { init };
})();
