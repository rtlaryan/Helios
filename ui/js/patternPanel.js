/**
 * patternPanel.js — Radiation pattern: az/el Chart.js cuts + 3D Three.js lobe.
 */

const PatternPanel = (() => {
    let azChart = null, elChart = null;
    let patternScene, patternCamera, patternRenderer, patternControls, patternMesh;
    let geoScene, geoCamera, geoRenderer, geoControls;

    // ── Chart.js helpers ───────────────────────────────────────────────────────
    const commonScales = (xMin, xMax, xLabel) => ({
        x: {
            type: 'linear',
            min: xMin,
            max: xMax,
            grid: { color: 'rgba(99,179,237,0.08)' },
            ticks: {
                color: '#4a6280',
                font: { family: 'JetBrains Mono', size: 10 },
                maxTicksLimit: 11,
                callback(v) { return v === 0 ? '0' : v.toFixed(0); },
            },
            title: { display: true, text: xLabel, color: '#4a6280', font: { size: 10 } },
        },
        y: {
            min: -80, max: 5,
            grid: { color: 'rgba(99,179,237,0.07)' },
            ticks: {
                color: '#4a6280',
                font: { family: 'JetBrains Mono', size: 10 },
                callback: v => v + ' dB',
            },
        },
    });

    const commonOptions = (xMin, xMax, xLabel) => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        parsing: false, // data is already in {x, y} form
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: { label: ctx => ctx.parsed.x.toFixed(1) + '°  →  ' + ctx.parsed.y.toFixed(1) + ' dB' },
                bodyFont: { family: 'JetBrains Mono', size: 11 },
            },
        },
        scales: commonScales(xMin, xMax, xLabel),
    });

    function makeChart(canvasId, xMin, xMax, xLabel) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;
        return new Chart(canvas, {
            type: 'line',
            data: {
                datasets: [{
                    data: [],
                    borderColor: '#3b9eff',
                    backgroundColor: 'rgba(59,158,255,0.07)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.2,
                }],
            },
            options: commonOptions(xMin, xMax, xLabel),
        });
    }

    /** Downsample to ≤ maxPts {x, y} points for rendering speed. */
    function toXY(axisArr, respArr, maxPts = 400) {
        const step = Math.max(1, Math.floor(axisArr.length / maxPts));
        const pts  = [];
        for (let i = 0; i < axisArr.length; i += step) {
            pts.push({ x: axisArr[i], y: respArr[i] });
        }
        return pts;
    }

    function updateCharts(data) {
        if (!azChart) azChart = makeChart('az-chart', -180, 180, 'Azimuth (°)');
        if (!elChart) elChart = makeChart('el-chart',  -90,  90, 'Elevation (°)');

        azChart.data.datasets[0].data = toXY(data.az_axis_deg, data.az_response_db);
        azChart.update('none');

        elChart.data.datasets[0].data = toXY(data.el_axis_deg, data.el_response_db);
        elChart.update('none');
    }

    // ── Shared axis-line helper ────────────────────────────────────────────────
    /** Add X (red), Y (green), Z (blue) axis lines at the given length to a scene. */
    function addAxisLines(targetScene, length = 1.2) {
        const axMat  = c => new THREE.LineBasicMaterial({ color: c, opacity: 0.4, transparent: true });
        const axLine = (from, to, c) => {
            const g = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(...from), new THREE.Vector3(...to),
            ]);
            return new THREE.Line(g, axMat(c));
        };
        targetScene.add(axLine([0, 0, 0], [length, 0, 0],      0xf87171));
        targetScene.add(axLine([0, 0, 0], [0, length, 0],      0x34d399));
        targetScene.add(axLine([0, 0, 0], [0, 0, length],      0x3b9eff));
    }

    // ── 3D pattern in Three.js ─────────────────────────────────────────────────
    function init3DPattern(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || patternRenderer) return;

        patternRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        patternRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        patternRenderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
        patternRenderer.setClearColor(0x070b14, 1);

        patternScene  = new THREE.Scene();
        patternCamera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.001, 100);
        patternCamera.position.set(1.5, 1, 1.5);
        patternCamera.lookAt(0, 0, 0);

        patternScene.add(new THREE.AmbientLight(0xffffff, 0.5));
        const dlight = new THREE.DirectionalLight(0xffffff, 0.8);
        dlight.position.set(2, 2, 2);
        patternScene.add(dlight);

        addAxisLines(patternScene, 1.2);

        patternControls = new THREE.OrbitControls(patternCamera, canvas);
        patternControls.enableDamping = true;
        patternControls.dampingFactor = 0.08;

        const resizeObs = new ResizeObserver(() => {
            patternRenderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
            patternCamera.aspect = canvas.clientWidth / canvas.clientHeight;
            patternCamera.updateProjectionMatrix();
        });
        resizeObs.observe(canvas.parentElement);

        (function loop() {
            requestAnimationFrame(loop);
            patternControls?.update();
            patternRenderer.render(patternScene, patternCamera);
        })();
    }

    function update3DPattern(data) {
        if (!patternScene) return;
        if (patternMesh) { patternScene.remove(patternMesh); patternMesh = null; }

        const p3d  = data.pattern3d;
        const { X, Y, Z, C } = p3d;
        const rows = X.length, cols = X[0].length;

        const geo       = new THREE.BufferGeometry();
        const positions = [], colors = [];
        const floorDb   = -40;

        /** Map a dB value to an [r, g, b] triple in [0, 1]. */
        const dbToColor = (db) => {
            const t = Math.max(0, Math.min(1, (db - floorDb) / (-floorDb)));
            return [
                (t < 0.5 ? 0 : (t - 0.5) * 2 * 59 + (1 - (t - 0.5) * 2) * 167) / 255,
                139 * t / 255,
                (250 - t * 210) / 255,
            ];
        };

        for (let r = 0; r < rows - 1; r++) {
            for (let c = 0; c < cols - 1; c++) {
                // Two triangles per quad: (r,c)→(r+1,c)→(r,c+1) and (r+1,c)→(r+1,c+1)→(r,c+1)
                const vs = [
                    [X[r][c],     Z[r][c],     Y[r][c]],
                    [X[r+1][c],   Z[r+1][c],   Y[r+1][c]],
                    [X[r][c+1],   Z[r][c+1],   Y[r][c+1]],
                    [X[r+1][c],   Z[r+1][c],   Y[r+1][c]],
                    [X[r+1][c+1], Z[r+1][c+1], Y[r+1][c+1]],
                    [X[r][c+1],   Z[r][c+1],   Y[r][c+1]],
                ];
                const dbs = [C[r][c], C[r+1][c], C[r][c+1], C[r+1][c], C[r+1][c+1], C[r][c+1]];
                vs.forEach((v, i) => {
                    positions.push(...v);
                    colors.push(...dbToColor(dbs[i]));
                });
            }
        }

        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
        geo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(colors),    3));
        geo.computeVertexNormals();

        patternMesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
            vertexColors: true, side: THREE.DoubleSide,
            transparent: true, opacity: 0.85, shininess: 30,
        }));
        patternScene.add(patternMesh);
    }

    // ── Array geometry 3D ──────────────────────────────────────────────────────
    function init3DGeo(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || geoRenderer) return;

        geoRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        geoRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        geoRenderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
        geoRenderer.setClearColor(0x070b14, 1);

        geoScene  = new THREE.Scene();
        geoCamera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.0001, 100);
        geoCamera.position.set(0, 0.08, 0.2);
        geoCamera.lookAt(0, 0, 0);

        geoScene.add(new THREE.AmbientLight(0xffffff, 0.8));

        geoControls = new THREE.OrbitControls(geoCamera, canvas);
        geoControls.enableDamping = true;
        geoControls.dampingFactor = 0.08;
        geoControls.target.set(0, 0, 0);

        const resizeObs = new ResizeObserver(() => {
            geoRenderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
            geoCamera.aspect = canvas.clientWidth / canvas.clientHeight;
            geoCamera.updateProjectionMatrix();
        });
        resizeObs.observe(canvas.parentElement);

        (function loop() {
            requestAnimationFrame(loop);
            geoControls?.update();
            geoRenderer.render(geoScene, geoCamera);
        })();
    }

    function updateArrayGeo(data) {
        if (!geoScene) return;

        // Clear old objects
        while (geoScene.children.length) geoScene.remove(geoScene.children[0]);
        geoScene.add(new THREE.AmbientLight(0xffffff, 0.8));

        const pos    = data.positions_mm;
        const amp    = data.weights.amplitude;
        const phase  = data.weights.phase;
        const N      = amp.length;

        // Normalise positions to fit within ±0.15 units
        let maxCoord = 0;
        for (let i = 0; i < N; i++) maxCoord = Math.max(maxCoord, Math.abs(pos.y[i]), Math.abs(pos.z[i]));
        const sc     = maxCoord > 0 ? 0.15 / maxCoord : 1.0;
        const maxAmp = Math.max(...amp);

        const geo       = new THREE.BufferGeometry();
        const positions = [], colors = [];
        for (let i = 0; i < N; i++) {
            positions.push(pos.x[i] * sc, pos.z[i] * sc, pos.y[i] * sc);
            const hue = (phase[i] + Math.PI) / (2 * Math.PI);
            const c   = new THREE.Color().setHSL(hue, 0.9, 0.6);
            colors.push(c.r, c.g, c.b);
        }
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
        geo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(colors), 3));

        geoScene.add(new THREE.Points(geo, new THREE.PointsMaterial({
            size: 0.015, vertexColors: true, sizeAttenuation: true,
        })));

        addAxisLines(geoScene, 0.2);
    }

    // ── Main compute ───────────────────────────────────────────────────────────
    async function computePattern() {
        if (!HeliosState.antennaBatch) {
            toast('Load an antenna first (Antenna tab)', 'error');
            return;
        }

        const btn         = document.getElementById('btn-compute-pattern');
        const progressWrap = document.getElementById('pattern-progress');
        const progressBar  = document.getElementById('pattern-progress-bar');
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div> Computing…';
        if (progressWrap) { progressWrap.style.display = ''; progressBar.style.width = '0%'; }

        try {
            // Animate a fake progress bar for UX feedback
            const bumpProgress = setInterval(() => {
                const current = parseFloat(progressBar.style.width) || 0;
                if (current < 80) progressBar.style.width = (current + Math.random() * 15) + '%';
            }, 300);

            const resolution = parseInt(document.getElementById('pattern-resolution')?.value || 150);
            const data = await HeliosAPI.computePattern(null, resolution); // reuse cached batch
            HeliosState.patternData = data;

            clearInterval(bumpProgress);
            if (progressBar) progressBar.style.width = '100%';

            ['card-az-cut', 'card-el-cut', 'card-3d-pattern', 'card-array-geo'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = '';
            });

            updateCharts(data);

            // Give DOM time to render canvases before initialising WebGL
            setTimeout(() => {
                init3DPattern('array-geo-canvas');
                update3DPattern(data);
                init3DGeo('geo-canvas-small');
                updateArrayGeo(HeliosState.antennaBatch);
            }, 50);

            // Optional ground projection
            if (document.getElementById('pattern-overlay')?.value === 'show') {
                try {
                    const proj = await HeliosAPI.groundProjection(2.0);
                    HeliosState.groundProjection = proj;
                    GlobeRenderer.updateGroundProjection(proj.db_map, proj.lat_vec, proj.lon_vec);
                } catch (_) { /* non-critical */ }
            }

            toast('Pattern computed', 'success');
        } catch (err) {
            toast('Pattern computation failed: ' + err.message, 'error');
            console.error(err);
        } finally {
            if (progressWrap) setTimeout(() => { progressWrap.style.display = 'none'; }, 500);
            btn.disabled = false;
            btn.innerHTML = `
                <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polygon points="5,3 19,12 5,21 5,3"/></svg>
                Compute Pattern`;
        }
    }

    function init() {
        document.getElementById('btn-compute-pattern')?.addEventListener('click', computePattern);

        HeliosState.on('antennaLoaded', () => {
            if (geoScene && HeliosState.antennaBatch) updateArrayGeo(HeliosState.antennaBatch);
        });
    }

    return { init };
})();
