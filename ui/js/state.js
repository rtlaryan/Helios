/**
 * state.js — Shared application state (singleton).
 * All modules read/write this object directly.
 */
window.HeliosState = {
    // Drawn zones
    zones: [],          // [{id, type, lat, lon, radius_deg, peak_db, rolloff, color, mesh}]

    // Current draw mode
    drawMode: 'circle', // 'circle' | 'polygon'
    zoneType: 'importance', // 'power' | 'importance'

    // Zone defaults from sliders
    zoneParams: {
        peak_db: 0,
        rolloff: 5,
        radius_deg: 8,
    },

    // Generated maps
    maps: null,         // { lat_vec, lon_vec, power_map, importance_map, shape }
    mapsResolution: 0.5,
    overlayMode: 'power',
    overlayOpacity: 0.75,

    // Antenna / batch data
    antennaBatch: null, // JSON from /api/antenna

    // Pattern data
    patternData: null,  // JSON from /api/pattern

    // Ground projection
    groundProjection: null,

    // Active header view
    activeView: 'target',

    // Event bus
    _listeners: {},
    on(event, cb) {
        (this._listeners[event] = this._listeners[event] || []).push(cb);
    },
    emit(event, data) {
        (this._listeners[event] || []).forEach(cb => cb(data));
    },
};

// ── Toast helper ──────────────────────────────────────────────────────────────
window.toast = function (msg, type = 'info', durationMs = 3000) {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transition = 'opacity 0.3s';
        setTimeout(() => el.remove(), 300);
    }, durationMs);
};

// ── Sidebar tab switching ──────────────────────────────────────────────────────
document.querySelectorAll('.stab').forEach(tab => {
    tab.addEventListener('click', () => {
        const target = tab.dataset.tab;
        document.querySelectorAll('.stab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(target)?.classList.add('active');
    });
});

// ── Collapsible cards ──────────────────────────────────────────────────────────
document.querySelectorAll('.card-header').forEach(header => {
    header.addEventListener('click', () => {
        header.closest('.card')?.classList.toggle('collapsed');
    });
});

// ── Slider live display update ────────────────────────────────────────────────
function bindSlider(id, displayId, fmt) {
    const el   = document.getElementById(id);
    const disp = document.getElementById(displayId);
    if (!el || !disp) return;
    const update = () => { disp.textContent = fmt(el.value); };
    el.addEventListener('input', update);
    update();
}
bindSlider('zone-db',       'zone-db-val',       v => `${v} dB`);
bindSlider('zone-rolloff',  'zone-rolloff-val',  v => `${v}°`);
bindSlider('zone-radius',   'zone-radius-val',   v => `${v}°`);
bindSlider('overlay-opacity','overlay-opacity-val', v => `${Math.round(v * 100)}%`);
bindSlider('pat-el-cut',    'pat-el-cut-val',    v => `${v}°`);

// Sync zone sliders → HeliosState.zoneParams
['zone-db', 'zone-rolloff', 'zone-radius'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => {
        HeliosState.zoneParams = {
            peak_db:    parseFloat(document.getElementById('zone-db').value),
            rolloff:    parseFloat(document.getElementById('zone-rolloff').value),
            radius_deg: parseFloat(document.getElementById('zone-radius').value),
        };
        drawRolloffPreview();
    });
});

// ── Rolloff (Gaussian) preview canvas ──────────────────────────────────────────
function drawRolloffPreview() {
    const canvas = document.getElementById('rolloff-preview');
    if (!canvas) return;
    const W = canvas.offsetWidth || 220, H = canvas.offsetHeight || 56;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const { peak_db, rolloff, radius_deg } = HeliosState.zoneParams;
    const sigma = Math.max(radius_deg * rolloff, 0.01);
    const xMax  = radius_deg * 2.2;  // show slightly past the radius
    const floor = -60;

    // Compute dB value at a symmetric x position
    const valDb = x => peak_db + 10 * Math.log10(Math.exp(-(x * x) / (2 * sigma * sigma)));
    const toY   = db => H - H * Math.max(0, (db - floor) / (-floor));

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(59,158,255,0.55)');
    grad.addColorStop(1, 'rgba(59,158,255,0.04)');

    ctx.beginPath();
    for (let px = 0; px < W; px++) {
        const x = (px / W) * xMax - xMax / 2;
        px === 0 ? ctx.moveTo(px, toY(valDb(x))) : ctx.lineTo(px, toY(valDb(x)));
    }
    ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Curve stroke
    ctx.beginPath();
    for (let px = 0; px < W; px++) {
        const x = (px / W) * xMax - xMax / 2;
        px === 0 ? ctx.moveTo(px, toY(valDb(x))) : ctx.lineTo(px, toY(valDb(x)));
    }
    ctx.strokeStyle = '#3b9eff';
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    // Zone radius marker lines
    const rPx = (radius_deg / xMax) * W;
    const cx  = W / 2;
    [cx - rPx, cx + rPx].forEach(rx => {
        ctx.beginPath(); ctx.moveTo(rx, 0); ctx.lineTo(rx, H);
        ctx.strokeStyle = 'rgba(251,191,36,0.55)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.stroke(); ctx.setLineDash([]);
    });

    // Labels
    ctx.fillStyle = '#8fa3bf'; ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillText(`${peak_db >= 0 ? '+' : ''}${peak_db} dB`, 4, 12);
    ctx.fillText(`σ=${rolloff}°`, 4, H - 4);
}

document.addEventListener('DOMContentLoaded', () => {
    drawRolloffPreview();

    // Focus Region toggle
    document.getElementById('focus-header')?.addEventListener('click', e => {
        if (e.target.closest('#focus-enable-label')) return;
        const body = document.getElementById('focus-body');
        body.style.display = body.style.display === 'none' ? '' : 'none';
    });

    // Auto-fit focus region from drawn zones
    document.getElementById('btn-focus-from-zones')?.addEventListener('click', () => {
        const zones = HeliosState.zones;
        if (!zones.length) { toast('No zones drawn yet', 'error'); return; }

        let latMin = 90, latMax = -90, lonMin = 180, lonMax = -180;
        const pad = 5; // degrees padding

        for (const z of zones) {
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

        document.getElementById('focus-lat-min').value = Math.max(-90,  Math.floor(latMin - pad));
        document.getElementById('focus-lat-max').value = Math.min( 90,  Math.ceil(latMax  + pad));
        document.getElementById('focus-lon-min').value = Math.max(-180, Math.floor(lonMin - pad));
        document.getElementById('focus-lon-max').value = Math.min( 180, Math.ceil(lonMax  + pad));
        toast('Focus region set from zones', 'success');
    });
});

// ── Zone type toggle ───────────────────────────────────────────────────────────
document.querySelectorAll('.zone-type-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.zone-type-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        HeliosState.zoneType = btn.dataset.type;
    });
});

// ── Header tab switching ───────────────────────────────────────────────────────
function activateSidebarTab(tabId) {
    if (!tabId) return;
    document.querySelectorAll('.stab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`.stab[data-tab="${tabId}"]`)?.classList.add('active');
    document.getElementById(tabId)?.classList.add('active');
}

document.querySelectorAll('.htab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.htab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        HeliosState.activeView = tab.dataset.view;

        if (tab.dataset.sidebarTab) activateSidebarTab(tab.dataset.sidebarTab);

        const toolbar     = document.getElementById('draw-toolbar');
        const zoneOverlay = document.getElementById('zone-list-overlay');
        const isTarget    = HeliosState.activeView === 'target';
        if (toolbar)     toolbar.style.display     = isTarget ? '' : 'none';
        if (zoneOverlay) zoneOverlay.style.display = isTarget ? '' : 'none';

        HeliosState.emit('viewChanged', HeliosState.activeView);
    });
});

// ── Auto-rotate button ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const btn       = document.getElementById('btn-autorotate');
    const iconPlay  = document.getElementById('autorotate-icon-play');
    const iconPause = document.getElementById('autorotate-icon-pause');
    const label     = document.getElementById('autorotate-label');
    if (!btn) return;

    btn.addEventListener('click', () => {
        const nowRotating = GlobeRenderer.isAutoRotating();
        GlobeRenderer.setAutoRotate(!nowRotating);

        if (nowRotating) {
            iconPlay.style.display  = 'none';
            iconPause.style.display = '';
            label.textContent = 'Paused';
            btn.style.color = 'var(--warning)';
        } else {
            iconPlay.style.display  = '';
            iconPause.style.display = 'none';
            label.textContent = 'Rotating';
            btn.style.color = '';
        }
    });
});
