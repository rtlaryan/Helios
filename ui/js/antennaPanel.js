/**
 * antennaPanel.js — Antenna configuration, loading, and orbit visualization.
 */

const AntennaPanel = (() => {

    function getSpecParams() {
        return {
            lat: parseFloat(document.getElementById('ant-lat').value),
            lon: parseFloat(document.getElementById('ant-lon').value),
            alt: parseFloat(document.getElementById('ant-alt').value) * 1000, // km→m
            frequency: parseFloat(document.getElementById('ant-freq').value) * 1e9, // GHz→Hz
            spacing_ratio: parseFloat(document.getElementById('ant-spacing').value),
            element_count: parseInt(document.getElementById('ant-elements').value),
            geometry: document.getElementById('ant-geometry').value,
            aspect_ratio: parseFloat(document.getElementById('ant-aspect').value),
        };
    }

    function updateStats(data) {
        const statsEl = document.getElementById('antenna-stats');
        if (statsEl) statsEl.style.display = 'grid';
        document.getElementById('stat-ant-elements').textContent = data.element_count;
        document.getElementById('stat-ant-wl').textContent = (data.wavelength_m * 100).toFixed(2) + ' cm';
        document.getElementById('stat-ant-alt').textContent = (data.lla.alt / 1000).toFixed(0) + ' km';
        document.getElementById('stat-ant-freq').textContent = data.frequency_ghz + ' GHz';
    }

    async function loadAntenna() {
        const btn = document.getElementById('btn-load-antenna');
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div> Loading…';

        try {
            const params = getSpecParams();
            const data = await HeliosAPI.loadAntenna(params);
            HeliosState.antennaBatch = data;

            // Show on globe
            GlobeRenderer.setAntennaPosition(data.lla.lat, data.lla.lon, data.lla.alt / 1000);
            updateStats(data);

            toast(`Loaded ${data.element_count}-element ${params.geometry} array`, 'success');
            HeliosState.emit('antennaLoaded', data);
        } catch (err) {
            toast('Failed to load antenna: ' + err.message, 'error');
            console.error(err);
        } finally {
            btn.disabled = false;
            btn.innerHTML = `
        <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 1 0 0 14.14"/><path d="M22 2l-5 5"/></svg>
        Load Antenna`;
        }
    }

    function init() {
        document.getElementById('btn-load-antenna')?.addEventListener('click', loadAntenna);

        // Manual batch import via file picker
        document.getElementById('btn-import-batch')?.addEventListener('click', async () => {
            const fileInput = document.getElementById('batch-import-file');
            if (!fileInput.files.length) { toast('Choose a JSON file first', 'error'); return; }
            const btn = document.getElementById('btn-import-batch');
            btn.disabled = true; btn.textContent = '…';
            try {
                const text = await fileInput.files[0].text();
                const json = JSON.parse(text);
                const data = await HeliosAPI.importBatch(json);
                HeliosState.antennaBatch = data;
                GlobeRenderer.setAntennaPosition(data.lla.lat, data.lla.lon, data.lla.alt / 1000);
                updateStats(data);
                toast(`Imported ${data.element_count}-element batch`, 'success');
                HeliosState.emit('antennaLoaded', data);
            } catch (err) {
                toast('Import failed: ' + err.message, 'error');
                console.error(err);
            } finally {
                btn.disabled = false; btn.textContent = 'Import';
            }
        });
    }

    return { init, getSpecParams };
})();
