/**
 * globe.js — Three.js 3D Earth globe with OrbitControls.
 *
 * Public API:
 *   GlobeRenderer.init(canvasId)
 *   GlobeRenderer.latLonToXYZ(lat, lon, r)
 *   GlobeRenderer.setAntennaPosition(lat, lon, altKm)
 *   GlobeRenderer.setOrbitEnabled(enabled)
 *   GlobeRenderer.setAutoRotate(enabled)
 *   GlobeRenderer.isAutoRotating()
 *   GlobeRenderer.zoomCamera(factor)
 *   GlobeRenderer.rayCastLatLon(mouseEvent) → {lat, lon} | null
 *   GlobeRenderer.addZoneMesh(lat, lon, radiusDeg, color, opacity) → id
 *   GlobeRenderer.removeZoneMesh(id)
 *   GlobeRenderer.addPolygonZoneMesh(verts, color, opacity) → id
 *   GlobeRenderer.addPolyPreviewLine(verts) → line
 *   GlobeRenderer.removePolyPreview(line)
 *   GlobeRenderer.updateMapOverlay(maps, mode, opacity)
 *   GlobeRenderer.updateGroundProjection(dbMap, latVec, lonVec)
 */

const GlobeRenderer = (() => {
    const EARTH_R = 1.0;
    let scene, camera, renderer, controls;
    let earthMesh, starField;
    let overlayMeshPower = null, overlayMeshImportance = null;
    let antennaMesh = null;
    let groundProjectionMesh = null;
    const zoneMeshes = {}; // id → mesh
    let _autoRotate = false;
    let _focusRegionLines = null;

    // ── Lat/Lon ↔ XYZ ─────────────────────────────────────────────────────────
    function latLonToXYZ(lat, lon, r = EARTH_R) {
        const phi   = (90 - lat) * (Math.PI / 180);
        const theta = (lon + 180) * (Math.PI / 180);
        return new THREE.Vector3(
            -r * Math.sin(phi) * Math.cos(theta),
             r * Math.cos(phi),
             r * Math.sin(phi) * Math.sin(theta),
        );
    }

    // ── Photoreal Earth textures (NASA Blue Marble via CDN) ───────────────────
    const EARTH_TEX_URL = 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg';
    const BUMP_URL      = 'https://unpkg.com/three-globe/example/img/earth-topology.png';
    const SPEC_URL      = 'https://unpkg.com/three-globe/example/img/earth-water.png';

    /** Procedural fallback texture — used while CDN loads or on failure. */
    function makeProceduralTexture() {
        const w = 1024, h = 512;
        const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
        const ctx = cv.getContext('2d');
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, '#0a2040'); grad.addColorStop(0.5, '#0d3060'); grad.addColorStop(1, '#0a2040');
        ctx.fillStyle = grad; ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = '#1e4d2b';
        [[100,200,320,180],[520,150,260,160],[700,250,200,140],[400,320,180,100]].forEach(([x,y,rw,rh]) => {
            ctx.beginPath(); ctx.ellipse(x, y, rw, rh, 0, 0, Math.PI * 2); ctx.fill();
        });
        return new THREE.CanvasTexture(cv);
    }

    function loadEarthTexture(material) {
        const loader = new THREE.TextureLoader();
        loader.setCrossOrigin('anonymous');
        loader.load(EARTH_TEX_URL,
            tex => { material.map = tex; material.needsUpdate = true; },
            undefined,
            ()  => { material.map = makeProceduralTexture(); material.needsUpdate = true; }
        );
        loader.load(BUMP_URL, tex => { material.bumpMap = tex; material.bumpScale = 0.008; material.needsUpdate = true; });
        loader.load(SPEC_URL, tex => { material.specularMap = tex; material.needsUpdate = true; });
    }

    // ── Star field ─────────────────────────────────────────────────────────────
    function makeStarField() {
        const N = 4000;
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(N * 3);
        for (let i = 0; i < N * 3; i++) pos[i] = (Math.random() - 0.5) * 200;
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        return new THREE.Points(geo, new THREE.PointsMaterial({
            color: 0xffffff, size: 0.12, transparent: true, opacity: 0.7,
        }));
    }

    // ── Atmosphere glow ────────────────────────────────────────────────────────
    function makeAtmosphere() {
        const geo = new THREE.SphereGeometry(EARTH_R * 1.015, 64, 64);
        const mat = new THREE.MeshPhongMaterial({
            color: 0x2244aa, side: THREE.BackSide, transparent: true, opacity: 0.08,
        });
        return new THREE.Mesh(geo, mat);
    }

    // ── Init ───────────────────────────────────────────────────────────────────
    function init(canvasId) {
        const canvas = document.getElementById(canvasId);

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.setClearColor(0x000000, 0);

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.01, 1000);
        camera.position.set(0, 0, 2.8);

        scene.add(new THREE.AmbientLight(0x334466, 0.8));
        const sun = new THREE.DirectionalLight(0xffeedd, 1.2);
        sun.position.set(5, 3, 5);
        scene.add(sun);
        const rim = new THREE.DirectionalLight(0x1144aa, 0.3);
        rim.position.set(-3, -1, -3);
        scene.add(rim);

        // Earth — start with procedural texture, swap to photoreal once CDN responds
        const earthMat = new THREE.MeshPhongMaterial({
            map: makeProceduralTexture(),
            specular: new THREE.Color(0x334455),
            shininess: 15,
        });
        earthMesh = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R, 128, 128), earthMat);
        scene.add(earthMesh);
        loadEarthTexture(earthMat);

        scene.add(makeAtmosphere());
        starField = makeStarField();
        scene.add(starField);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping  = true;
        controls.dampingFactor  = 0.08;
        controls.minDistance    = 1.02;
        controls.maxDistance    = 8.0;
        controls.enablePan      = false;
        controls.rotateSpeed    = 0.5;

        const resizeObserver = new ResizeObserver(() => {
            renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
        });
        resizeObserver.observe(canvas.parentElement);

        (function animate() {
            requestAnimationFrame(animate);
            controls.update();
            if (_autoRotate) {
                earthMesh.rotation.y += 0.0003;
                if (starField) starField.rotation.y += 0.00005;
            }
            renderer.render(scene, camera);
        })();
    }

    // ── Orbit controls ─────────────────────────────────────────────────────────
    function setOrbitEnabled(enabled) {
        if (controls) controls.enableRotate = enabled;
    }

    function zoomCamera(factor) {
        if (!controls || !camera) return;
        const newDist = camera.position.length() * factor;
        const dist = Math.max(controls.minDistance, Math.min(controls.maxDistance, newDist));
        camera.position.normalize().multiplyScalar(dist);
        controls.update();
    }

    // ── Auto-rotation ──────────────────────────────────────────────────────────
    function setAutoRotate(enabled) { _autoRotate = enabled; }
    function isAutoRotating() { return _autoRotate; }

    // ── Spherical zone disc (geodesic cap) ────────────────────────────────────
    let _zoneCounter = 0;

    function addZoneMesh(lat, lon, radiusDeg, color = 0xfbbf24, opacity = 0.35) {
        const id = ++_zoneCounter;
        const SURFACE_R  = EARTH_R * 1.006;
        const radiusRad  = radiusDeg * Math.PI / 180;
        const N          = 96;

        const up  = latLonToXYZ(lat, lon, 1.0).normalize();
        const ref = Math.abs(up.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
        const t1  = new THREE.Vector3().crossVectors(up, ref).normalize();
        const t2  = new THREE.Vector3().crossVectors(up, t1).normalize();

        // Geodesic ring point: p = cos(α)·n̂ + sin(α)·(cos(φ)·t1 + sin(φ)·t2)
        function ringPoint(phi) {
            return new THREE.Vector3()
                .addScaledVector(up, Math.cos(radiusRad))
                .addScaledVector(t1, Math.sin(radiusRad) * Math.cos(phi))
                .addScaledVector(t2, Math.sin(radiusRad) * Math.sin(phi))
                .normalize()
                .multiplyScalar(SURFACE_R);
        }

        // Fill: fan triangles from geodesic centre to ring
        const centre3   = up.clone().multiplyScalar(SURFACE_R);
        const positions = [];
        for (let i = 0; i < N; i++) {
            const a0 = (i / N) * 2 * Math.PI;
            const a1 = ((i + 1) / N) * 2 * Math.PI;
            positions.push(...centre3.toArray(), ...ringPoint(a0).toArray(), ...ringPoint(a1).toArray());
        }
        const fillGeo = new THREE.BufferGeometry();
        fillGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
        const fillMesh = new THREE.Mesh(fillGeo, new THREE.MeshBasicMaterial({
            color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false,
            polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
        }));
        fillMesh.renderOrder = 1;
        earthMesh.add(fillMesh);
        zoneMeshes[id] = fillMesh;

        // Outline ring
        const ringPoints = [];
        for (let i = 0; i <= N; i++) ringPoints.push(ringPoint((i / N) * 2 * Math.PI));
        const ring = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(ringPoints),
            new THREE.LineBasicMaterial({ color, opacity: 0.95, transparent: true })
        );
        ring.renderOrder = 2;
        earthMesh.add(ring);
        zoneMeshes[`${id}_ring`] = ring;

        return id;
    }

    function removeZoneMesh(id) {
        [id, `${id}_ring`].forEach(k => {
            if (zoneMeshes[k]) { earthMesh.remove(zoneMeshes[k]); delete zoneMeshes[k]; }
        });
    }

    // ── Polygon zone mesh ──────────────────────────────────────────────────────
    function addPolygonZoneMesh(verts, color = 0xa78bfa, opacity = 0.30) {
        const id       = ++_zoneCounter;
        const SURFACE_R = EARTH_R * 1.006;

        const pts3d = verts.map(v => latLonToXYZ(v.lat, v.lon, SURFACE_R));

        // Centroid
        const clat      = verts.reduce((s, v) => s + v.lat, 0) / verts.length;
        const clon      = verts.reduce((s, v) => s + v.lon, 0) / verts.length;
        const centroid3 = latLonToXYZ(clat, clon, SURFACE_R);

        // Fan triangulation from centroid to each edge
        const positions = [];
        const N = pts3d.length;
        for (let i = 0; i < N; i++) {
            const a = pts3d[i];
            const b = pts3d[(i + 1) % N];
            positions.push(...centroid3.toArray(), ...a.toArray(), ...b.toArray());
        }

        const fillGeo  = new THREE.BufferGeometry();
        fillGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
        const fillMesh = new THREE.Mesh(fillGeo, new THREE.MeshBasicMaterial({
            color, transparent: true, opacity,
            side: THREE.DoubleSide, depthWrite: false,
            polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
        }));
        fillMesh.renderOrder = 1;
        earthMesh.add(fillMesh);
        zoneMeshes[id] = fillMesh;

        // Outline: geodesic arc (8 interpolated points per edge)
        const outlinePts = [];
        for (let i = 0; i < N; i++) {
            const va = verts[i], vb = verts[(i + 1) % N];
            for (let t = 0; t <= 8; t++) {
                const f = t / 8;
                outlinePts.push(latLonToXYZ(
                    va.lat + (vb.lat - va.lat) * f,
                    va.lon + (vb.lon - va.lon) * f,
                    SURFACE_R * 1.002
                ));
            }
        }
        const ring = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(outlinePts),
            new THREE.LineBasicMaterial({ color, opacity: 0.95, transparent: true })
        );
        ring.renderOrder = 2;
        earthMesh.add(ring);
        zoneMeshes[`${id}_ring`] = ring;

        return id;
    }

    // ── Polygon preview line (during drawing) ─────────────────────────────────
    function addPolyPreviewLine(verts) {
        const SURFACE_R = EARTH_R * 1.008;
        const pts = verts.map(v => latLonToXYZ(v.lat, v.lon, SURFACE_R));
        pts.push(pts[0]); // close loop for visual feedback
        const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.7, transparent: true })
        );
        line.renderOrder = 3;
        earthMesh.add(line);
        return line;
    }

    function removePolyPreview(line) {
        if (line) earthMesh.remove(line);
    }

    // ── Focus region rectangle ─────────────────────────────────────────────────
    /**
     * Draw a dotted lat/lon bounding box on the globe surface.
     * Lines follow lat/lon grid lines (not great circles).
     */
    function updateFocusRegion(latMin, latMax, lonMin, lonMax) {
        clearFocusRegion();
        const R = EARTH_R * 1.009; // slightly above zone meshes
        const STEPS = 120;
        const pts = [];

        // Build a closed rectangle path: bottom edge → right edge → top edge → left edge
        const segments = [
            // bottom (latMin, lon from lonMin→lonMax)
            Array.from({length: STEPS + 1}, (_, i) => [latMin, lonMin + (lonMax - lonMin) * i / STEPS]),
            // right (lonMax, lat from latMin→latMax)
            Array.from({length: STEPS + 1}, (_, i) => [latMin + (latMax - latMin) * i / STEPS, lonMax]),
            // top (latMax, lon from lonMax→lonMin)
            Array.from({length: STEPS + 1}, (_, i) => [latMax, lonMax - (lonMax - lonMin) * i / STEPS]),
            // left (lonMin, lat from latMax→latMin)
            Array.from({length: STEPS + 1}, (_, i) => [latMax - (latMax - latMin) * i / STEPS, lonMin]),
        ];

        // Render each side as a separate dashed line (alternating visible segments)
        const DASH_DEG = 4; // approximate dash length in degrees of arc
        const group = new THREE.Group();

        for (const seg of segments) {
            // Split each side into alternating dash/gap segments
            let drawing = true;
            let dashBuf = [];
            let arcSoFar = 0;

            for (let k = 0; k < seg.length - 1; k++) {
                const [lat0, lon0] = seg[k];
                const [lat1, lon1] = seg[k + 1];
                const dlat = lat1 - lat0, dlon = lon1 - lon0;
                const segLen = Math.sqrt(dlat * dlat + dlon * dlon);
                arcSoFar += segLen;

                if (drawing) dashBuf.push(latLonToXYZ(lat0, lon0, R));

                if (arcSoFar >= DASH_DEG) {
                    if (drawing && dashBuf.length > 1) {
                        dashBuf.push(latLonToXYZ(lat1, lon1, R));
                        const line = new THREE.Line(
                            new THREE.BufferGeometry().setFromPoints(dashBuf),
                            new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.75, transparent: true })
                        );
                        line.renderOrder = 4;
                        group.add(line);
                    }
                    dashBuf = [];
                    arcSoFar = 0;
                    drawing = !drawing;
                }
            }
            // flush remaining dash
            if (drawing && dashBuf.length > 1) {
                const line = new THREE.Line(
                    new THREE.BufferGeometry().setFromPoints(dashBuf),
                    new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.75, transparent: true })
                );
                line.renderOrder = 4;
                group.add(line);
            }
        }

        _focusRegionLines = group;
        earthMesh.add(group);
    }

    function clearFocusRegion() {
        if (_focusRegionLines) { earthMesh.remove(_focusRegionLines); _focusRegionLines = null; }
    }

    // ── Antenna marker ─────────────────────────────────────────────────────────
    function setAntennaPosition(lat, lon, altKm) {
        if (antennaMesh) scene.remove(antennaMesh);

        const earthRadiusKm = 6371;
        const r   = EARTH_R * (1 + altKm / earthRadiusKm);
        const pos = latLonToXYZ(lat, lon, r);

        const group = new THREE.Group();

        // Satellite body
        group.add(new THREE.Mesh(
            new THREE.OctahedronGeometry(0.025, 0),
            new THREE.MeshPhongMaterial({ color: 0x3b9eff, emissive: 0x1144aa, shininess: 80 })
        ));

        // Nadir line
        const nadirDir = pos.clone().normalize().negate();
        const end = pos.clone().add(nadirDir.multiplyScalar(r - EARTH_R * 1.002));
        group.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), end.clone().sub(pos)]),
            new THREE.LineBasicMaterial({ color: 0x3b9eff, opacity: 0.3, transparent: true })
        ));

        group.position.copy(pos);
        earthMesh.add(group);
        antennaMesh = group;
    }

    // ── Shared canvas→texture→sphere overlay builder ───────────────────────────
    /**
     * Rasterize `grid` (2D array, row 0 = northernmost) into a canvas, blit it
     * at the correct lat/lon offset, then return a Three.js Mesh (sphere shell).
     *
     * @param {number[][]} grid       - 2D data array [H][W]
     * @param {number[]}   latVec     - latitude vector (ascending, length H)
     * @param {number[]}   lonVec     - longitude vector (ascending, length W)
     * @param {function}   colorFn    - (value) → [r, g, b, a]  (0-255 each)
     * @param {number}     opacity    - Three.js material opacity
     * @param {number}     radiusFactor - sphere radius as multiple of EARTH_R
     */
    function _buildSphereOverlay(grid, latVec, lonVec, colorFn, opacity, radiusFactor) {
        const H   = grid.length, W = grid[0].length;
        const res = Math.abs(latVec[1] - latVec[0]) || 0.5;
        const FW  = Math.round(360 / res);
        const FH  = Math.round(180 / res);

        const canvas = document.createElement('canvas');
        canvas.width = FW; canvas.height = FH;
        const ctx = canvas.getContext('2d');

        const img = ctx.createImageData(W, H);
        // latVec is ascending (south→north), but texture row 0 = north, so flip row index
        for (let r = 0; r < H; r++) {
            for (let c = 0; c < W; c++) {
                const [rv, gv, bv, av] = colorFn(grid[H - 1 - r][c]);
                const idx = (r * W + c) * 4;
                img.data[idx]     = rv;
                img.data[idx + 1] = gv;
                img.data[idx + 2] = bv;
                img.data[idx + 3] = av;
            }
        }
        const colOff = Math.round((lonVec[0] - (-180)) / res);
        const rowOff = Math.max(0, Math.round((90 - latVec[H - 1]) / res));
        ctx.putImageData(img, colOff, rowOff);

        const tex = new THREE.CanvasTexture(canvas);
        const geo = new THREE.SphereGeometry(EARTH_R * radiusFactor, 128, 64);
        const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity, depthWrite: false });
        return new THREE.Mesh(geo, mat);
    }

    // ── Ground projection overlay ──────────────────────────────────────────────
    function updateGroundProjection(dbMap, latVec, lonVec, minDb = -60, maxDb = 0) {
        if (groundProjectionMesh) earthMesh.remove(groundProjectionMesh);
        if (!dbMap) { groundProjectionMesh = null; return; }

        const viridisColor = (db) => {
            const t = Math.max(0, Math.min(1, (db - minDb) / (maxDb - minDb)));
            const r = Math.round(Math.min(255, Math.max(0, 68 + t * (94 - 68) + t * t * (253 - 94) - t * t * t * 253)));
            const g = Math.round(Math.min(255, Math.max(0, 1  + t * (198 - 1)  + t * t * (231 - 198))));
            const b = Math.round(Math.min(255, Math.max(0, 84 + t * (47 - 84)  + t * t * (37 - 47))));
            const a = db < minDb + 5 ? 0 : Math.round(t * 160 + 40);
            return [r, g, b, a];
        };

        groundProjectionMesh = _buildSphereOverlay(dbMap, latVec, lonVec, viridisColor, 0.65, 1.004);
        earthMesh.add(groundProjectionMesh);
    }

    // ── Map overlay (power / importance) ──────────────────────────────────────
    function updateMapOverlay(maps, mode, opacity) {
        if (overlayMeshPower)      { earthMesh.remove(overlayMeshPower);      overlayMeshPower      = null; }
        if (overlayMeshImportance) { earthMesh.remove(overlayMeshImportance); overlayMeshImportance = null; }
        if (!maps || mode === 'none') return;

        const isLinear = maps.power_normalized;
        const powerColor = isLinear
            ? (v) => {
                // Linear [0,1] — map directly
                if (v < 0.01) return [0, 0, 0, 0];
                const t = Math.max(0, Math.min(1, v));
                const r = Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 255);
                const g = Math.round(t < 0.5 ? t * 2 * 200 : 200 - (t - 0.5) * 2 * 100);
                const b = Math.round(t < 0.3 ? 200 : Math.max(0, 200 - (t - 0.3) / 0.7 * 200));
                return [r, g, b, Math.round(t * 200)];
            }
            : (db) => {
                // dB scale [-60, 0]
                const t = Math.max(0, Math.min(1, (db + 60) / 60));
                if (t < 0.05) return [0, 0, 0, 0];
                const r = Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 255);
                const g = Math.round(t < 0.5 ? t * 2 * 200 : 200 - (t - 0.5) * 2 * 100);
                const b = Math.round(t < 0.3 ? 200 : Math.max(0, 200 - (t - 0.3) / 0.7 * 200));
                return [r, g, b, Math.round(t * 200)];
            };

        const impColor = (v) => {
            if (v < 0.02) return [0, 0, 0, 0];
            return [
                Math.round(v * 167),
                Math.round(v * 139),
                Math.round(50 + v * 205),
                Math.round(v * 200),
            ];
        };

        if (mode === 'power' || mode === 'both') {
            overlayMeshPower = _buildSphereOverlay(maps.power_map, maps.lat_vec, maps.lon_vec, powerColor, opacity, 1.003);
            earthMesh.add(overlayMeshPower);
        }
        if (mode === 'importance' || mode === 'both') {
            overlayMeshImportance = _buildSphereOverlay(maps.importance_map, maps.lat_vec, maps.lon_vec, impColor, opacity, 1.003);
            earthMesh.add(overlayMeshImportance);
        }
    }

    // ── Raycasting ─────────────────────────────────────────────────────────────
    function rayCastLatLon(event) {
        const canvas = renderer.domElement;
        const rect   = canvas.getBoundingClientRect();

        const clientX = event.touches ? event.touches[0].clientX : event.clientX;
        const clientY = event.touches ? event.touches[0].clientY : event.clientY;

        const x = ((clientX - rect.left) / rect.width)  *  2 - 1;
        const y = ((clientY - rect.top)  / rect.height) * -2 + 1;

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
        const intersects = raycaster.intersectObject(earthMesh);
        if (intersects.length > 0) {
            // Apply earthMesh inverse transform to get the un-rotated local point
            const point = earthMesh.worldToLocal(intersects[0].point.clone());
            const lat   = 90 - (Math.acos(point.y / EARTH_R) * 180 / Math.PI);
            const lon   = (Math.atan2(point.z, -point.x) * 180 / Math.PI) - 180;
            return { lat, lon };
        }
        return null;
    }

    return {
        init, latLonToXYZ, setOrbitEnabled, setAutoRotate, isAutoRotating, zoomCamera,
        addZoneMesh, removeZoneMesh, addPolygonZoneMesh,
        addPolyPreviewLine, removePolyPreview,
        setAntennaPosition, updateGroundProjection, updateMapOverlay, rayCastLatLon,
        updateFocusRegion, clearFocusRegion,
    };
})();
