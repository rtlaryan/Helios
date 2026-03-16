/**
 * app.js — Main application entry point. Initializes all modules.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize the 3D globe
    GlobeRenderer.init('globe-canvas');

    // 2. Target zone drawing tools
    TargetEditor.init('globe-canvas');

    // 3. Map generation & preview
    MapRenderer.init();

    // 4. Antenna panel
    AntennaPanel.init();

    // 5. Pattern panel
    PatternPanel.init();

    // 6. Array geometry panel (header tab)
    ArrayGeoPanel.init();

    console.log('%cHelios Beamforming Studio', 'color:#3b9eff;font-size:16px;font-weight:bold;');
    console.log('%cReady. Server: http://localhost:5050', 'color:#8fa3bf;');
});
