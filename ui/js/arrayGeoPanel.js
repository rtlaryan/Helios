/**
 * arrayGeoPanel.js — Standalone array geometry tab panel (Array Geometry header tab).
 * Renders element positions + weights in a dedicated Three.js canvas when the user
 * navigates to the "Array Geometry" header tab.
 */

const ArrayGeoPanel = (() => {
    let scene, camera, renderer, controls, pointsMesh;
    let initialized = false;

    function init3D(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || initialized) return;
        initialized = true;

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
        renderer.setClearColor(0x070b14, 1);

        scene  = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.0001, 100);
        camera.position.set(0, 0.1, 0.3);

        scene.add(new THREE.AmbientLight(0xffffff, 0.9));

        controls = new THREE.OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;

        const resizeObs = new ResizeObserver(() => {
            renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
        });
        resizeObs.observe(canvas.parentElement);

        (function loop() {
            requestAnimationFrame(loop);
            controls.update();
            renderer.render(scene, camera);
        })();
    }

    function updateFromBatch(data) {
        if (!scene) return;

        // Clear scene, re-add light
        while (scene.children.length) scene.remove(scene.children[0]);
        scene.add(new THREE.AmbientLight(0xffffff, 0.9));

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
            const hue     = (phase[i] + Math.PI) / (2 * Math.PI);
            const ampNorm = amp[i] / maxAmp;
            const c = new THREE.Color().setHSL(hue, 0.9, 0.3 + ampNorm * 0.4);
            colors.push(c.r, c.g, c.b);
        }
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
        geo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(colors), 3));

        pointsMesh = new THREE.Points(geo, new THREE.PointsMaterial({
            size: 0.012, vertexColors: true, sizeAttenuation: true,
        }));
        scene.add(pointsMesh);

        // Axis lines
        const mkLine = (from, to, c) => {
            const g = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(...from), new THREE.Vector3(...to),
            ]);
            return new THREE.Line(g, new THREE.LineBasicMaterial({ color: c, opacity: 0.5, transparent: true }));
        };
        scene.add(mkLine([0, 0, 0], [0.2, 0,   0],   0xf87171));
        scene.add(mkLine([0, 0, 0], [0,   0.2, 0],   0x34d399));
        scene.add(mkLine([0, 0, 0], [0,   0,   0.2], 0x3b9eff));
    }

    // ── Array Geometry view (injected into main area when tab is active) ────────
    function buildArrayView() {
        const globeArea = document.getElementById('globe-area');
        let wrapper = document.getElementById('array-geo-view');
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.id = 'array-geo-view';
            wrapper.style.cssText = `
                position:absolute; inset:0; background:var(--bg-deep);
                display:flex; flex-direction:column; align-items:center; justify-content:center;
                gap:12px;
            `;

            const title = document.createElement('div');
            title.style.cssText = 'font-size:13px;font-weight:600;color:var(--text-sec);letter-spacing:0.06em;';
            title.textContent = 'ARRAY GEOMETRY & WEIGHTS';
            wrapper.appendChild(title);

            const hint = document.createElement('p');
            hint.id = 'array-geo-hint';
            hint.style.cssText = 'font-size:12px;color:var(--text-dim);text-align:center;';
            hint.textContent = 'Load an antenna in the Antenna tab to visualise its geometry.';
            wrapper.appendChild(hint);

            const canvas = document.createElement('canvas');
            canvas.id = 'array-geo-main-canvas';
            canvas.style.cssText = 'width:min(600px,90vw);height:min(500px,65vh);display:none;border-radius:12px;';
            wrapper.appendChild(canvas);

            const legend = document.createElement('div');
            legend.id = 'array-geo-legend';
            legend.style.cssText = 'display:none;gap:24px;font-size:11px;color:var(--text-dim);';
            legend.innerHTML = `
                <span style="color:var(--text-sec);">● Color = Phase angle</span>
                <span style="color:var(--text-sec);">● Size = Amplitude</span>
            `;
            wrapper.appendChild(legend);

            globeArea.appendChild(wrapper);
        }
        return wrapper;
    }

    function showArrayView(data) {
        const wrapper = buildArrayView();
        const canvas  = document.getElementById('array-geo-main-canvas');
        const hint    = document.getElementById('array-geo-hint');
        const legend  = document.getElementById('array-geo-legend');

        const hasData = Boolean(data);
        canvas.style.display          = hasData ? 'block' : 'none';
        if (legend) legend.style.display = hasData ? 'flex' : 'none';
        hint.style.display            = hasData ? 'none'  : '';
        wrapper.style.display         = 'flex';

        setTimeout(() => {
            init3D('array-geo-main-canvas');
            if (data) updateFromBatch(data);
        }, 30);
    }

    function hideArrayView() {
        const wrapper = document.getElementById('array-geo-view');
        if (wrapper) wrapper.style.display = 'none';
    }

    function init() {
        HeliosState.on('viewChanged', (view) => {
            if (view === 'array') {
                showArrayView(HeliosState.antennaBatch);
            } else {
                hideArrayView();
            }
        });

        HeliosState.on('antennaLoaded', (data) => {
            if (HeliosState.activeView === 'array') showArrayView(data);
            if (scene) updateFromBatch(data);
        });
    }

    return { init };
})();
