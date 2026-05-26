import { PHYSICS, API_DEFAULTS } from './constants.js';
import { SpaceSimulationAPI } from './api.js';
import { SpaceVisualization } from './visualization.js';
import { BrowserInference } from './browser_inference.js';
import { SpaceTycoon } from './tycoon.js';
import { TycoonUI } from './tycoon-ui.js';

let visualization = null;
let api = null;
let browserInference = null;
let browserReady = false;

// Tycoon
let tycoon = null;
let tycoonUI = null;
let tycoonActive = false;

const genTexts = [
    'Summoning planets...',
    'Stirring the cosmos...',
    'Whispering to the stars...',
    'Bending spacetime...',
    'Charming gravity...',
    'Dusting off nebula...',
];

document.addEventListener("DOMContentLoaded", async () => {
    api = new SpaceSimulationAPI();
    browserInference = new BrowserInference();
    visualization = new SpaceVisualization("canvas-container");

    // Check for saved tycoon game
    tycoon = new SpaceTycoon();
    tycoonActive = false;

    setupEventListeners();

    // Check for existing save to show tycoon mode option
    if (tycoon.hasSave()) {
        document.getElementById("mode-tycoon")?.classList.remove("locked");
    }

    await generateSystem();
});

function setupEventListeners() {
    // Generate button
    document.getElementById("generate-btn").addEventListener("click", async () => {
        await generateSystem();
    });

    // Pause button — shared between sim and tycoon
    document.getElementById("pause-btn").addEventListener("click", () => {
        if (tycoonActive && tycoon.active) {
            // Tycoon pause also pauses sim
            tycoon.togglePause();
            if (!tycoon.paused) {
                if (!visualization.isPlaying) visualization.play();
                setPauseIcon(true);
            } else {
                visualization.pause();
                setPauseIcon(false);
            }
        } else {
            if (visualization.isPlaying) {
                visualization.pause();
                setPauseIcon(false);
            } else {
                visualization.play();
                setPauseIcon(true);
            }
        }
    });

    // Speed slider
    const speedSlider = document.getElementById("speed-slider");
    const speedLabel = document.getElementById("speed-label");
    if (speedSlider) {
        speedSlider.addEventListener("input", () => {
            const m = parseFloat(speedSlider.value);
            visualization.setPlaybackSpeed(m);
            speedLabel.textContent = `${m}×`;
        });
    }

    // Mode toggle
    const modeSim = document.getElementById("mode-sim");
    const modeTycoon = document.getElementById("mode-tycoon");

    modeSim?.addEventListener("click", () => switchMode("sim"));
    modeTycoon?.addEventListener("click", () => switchMode("tycoon"));

    // Planet click handler — bridge to tycoon
    // The visualization's onMouseUp calls this.onBodyClick(e) with a mouse event.
    // We wrap it to add tycoon-specific behavior.
    const originalOnBodyClick = visualization.onBodyClick.bind(visualization);
    visualization.onBodyClick = function(event) {
        // Calculate which body was clicked (same logic as original)
        const rect = visualization.renderer.domElement.getBoundingClientRect();
        visualization.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        visualization.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        visualization.raycaster.setFromCamera(visualization.mouse, visualization.camera);
        const bodyMeshes = visualization.bodies.map(b => b.mesh);
        const intersects = visualization.raycaster.intersectObjects(bodyMeshes);

        if (intersects.length > 0) {
            const clickedMesh = intersects[0].object;
            const bodyIndex = visualization.bodies.findIndex(b => b.mesh === clickedMesh);

            if (bodyIndex !== -1 && bodyIndex > 0 && tycoonActive && tycoon.active) {
                // In tycoon mode, clicking a planet opens its panel
                tycoonUI.selectPlanet(bodyIndex);
                // Also select in visualization for the highlight ring
                if (visualization.selection.bodyIndex !== bodyIndex) {
                    visualization.selectBody(bodyIndex);
                }
                return;
            }
        }

        // Fall through to original behavior
        originalOnBodyClick(event);
    };
}

function switchMode(mode) {
    const modeSim = document.getElementById("mode-sim");
    const modeTycoon = document.getElementById("mode-tycoon");

    if (mode === "tycoon") {
        if (!tycoon.active && !tycoon.hasSave()) {
            toast("Generate a system first, then switch to Tycoon mode!");
            return;
        }
        tycoonActive = true;
        modeSim?.classList.remove("active");
        modeTycoon?.classList.add("active");
        document.body.classList.add("tycoon-active");

        if (!tycoon.active && tycoon.hasSave()) {
            // Load save
            tycoon.load();
            setupTycoonUI();
            tycoonUI.show();
            // Ensure sim is paused when loading tycoon
            if (visualization.isPlaying) {
                // Keep sim playing (orbits move) but tycoon is paused by default
            }
        } else {
            setupTycoonUI();
            tycoonUI.show();
        }

        toast("👑 Tycoon Mode — Click a planet to interact!");
    } else {
        tycoonActive = false;
        modeSim?.classList.add("active");
        modeTycoon?.classList.remove("active");
        document.body.classList.remove("tycoon-active");
        tycoonUI?.hide();
        tycoon.pause();
    }
}

function setupTycoonUI() {
    if (tycoonUI) tycoonUI.destroy();
    tycoonUI = new TycoonUI(tycoon);
    const container = document.getElementById("tycoon-container");
    if (container) {
        tycoonUI.create(container);
    }
    // Expose for cross-module access (planet list → visualization highlight)
    window._visualization = visualization;
}

function setPauseIcon(playing) {
    const btn = document.getElementById("pause-btn");
    btn.innerHTML = playing
        ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`
        : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
    btn.setAttribute("aria-label", playing ? "Pause" : "Play");
}

function toast(msg, duration = 2500) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(el._timeout);
    el._timeout = setTimeout(() => el.classList.add("hidden"), duration);
}

async function generateSystem() {
    const generateBtn = document.getElementById("generate-btn");
    generateBtn.disabled = true;

    const overlay = document.getElementById("generating-overlay");
    const genText = document.getElementById("generating-text");
    overlay.classList.remove("hidden");

    // If tycoon mode is active, end current game
    if (tycoonActive && tycoon.active) {
        tycoon.pause();
    }

    let textIndex = 0;
    const textInterval = setInterval(() => {
        textIndex = (textIndex + 1) % genTexts.length;
        genText.textContent = genTexts[textIndex];
    }, 3000);

    try {
        const centralMass = 0.1;
        const numBodies = Math.floor(Math.random() * 6) + 3;
        const temperature = 0.67;

        const safeMass = Math.max(centralMass, PHYSICS.CENTRAL_MASS_MIN);
        const simDt = PHYSICS.SIMULATION_DT_DEFAULT * (PHYSICS.MASS_RATIO / safeMass);

        let systemData;

        try {
            await warmBrowserModel();
            systemData = await browserInference.generateSystem(
                {
                    central_mass: centralMass,
                    num_bodies: numBodies,
                    temperature: temperature,
                    top_k: API_DEFAULTS.TOP_K,
                    simulation_timesteps: PHYSICS.SIMULATION_TIMESTEPS,
                    simulation_dt: simDt
                },
                {
                    onProgress: async (partialSystem, progress) => {
                        genText.textContent = `Received ${progress.completedPlanets}/${progress.totalPlanets} planets...`;
                        visualization.loadSystem(partialSystem);
                        updatePlanetCount(partialSystem);
                    }
                }
            );
        } catch (error) {
            console.warn("Browser inference failed, trying server:", error);
            genText.textContent = 'Contacting the observatory...';
            systemData = await api.generateSystem({
                central_mass: centralMass,
                num_bodies: numBodies,
                temperature: temperature,
                simulation_timesteps: PHYSICS.SIMULATION_TIMESTEPS,
                simulation_dt: simDt
            });
        }

        if (!systemData) {
            toast("No system returned — try again!");
            return;
        }

        visualization.loadSystem(systemData);
        updatePlanetCount(systemData);
        setPauseIcon(true);

        const trailModes = ['long', 'short'];
        const displayModes = ['both', 'planets-only', 'trails-only'];
        const trailMode = trailModes[Math.floor(Math.random() * trailModes.length)];
        const displayMode = displayModes[Math.floor(Math.random() * displayModes.length)];
        visualization.setTrailMode(trailMode);
        visualization.setDisplayMode(displayMode);

        const trailLabel = trailMode === 'long' ? 'streaming trails' : 'crisp orbits';
        const displayLabel = displayMode === 'both' ? '' : displayMode === 'planets-only' ? ' · naked planets' : ' · just trails';
        toast(`🪐 ${Math.max(0, systemData.bodies.length - 1)} planets · ${trailLabel}${displayLabel}`);

        // Initialize tycoon mode with the new system
        tycoon.deleteSave();
        tycoon.initFromSystem(systemData, visualization);
        tycoon.pause();

        // If we're in tycoon mode, update the UI
        if (tycoonActive) {
            setupTycoonUI();
            tycoonUI.show();
            setPauseIcon(false);
        }

    } catch (e) {
        console.error("Generation error:", e);
        toast("Something went wrong — try again!");
    } finally {
        clearInterval(textInterval);
        overlay.classList.add("hidden");
        generateBtn.disabled = false;
    }
}

function updatePlanetCount(systemData) {
    const count = Math.max(0, systemData.bodies.length - 1);
    const planetCountEl = document.getElementById("planet-count");
    if (tycoonActive && tycoon.active && tycoon.state) {
        const colonized = tycoon.state.planets.filter(p => p.colony).length;
        planetCountEl.textContent = `🪐 ${count} planets · 🏛 ${colonized} colonies`;
    } else {
        planetCountEl.textContent = `🪐 ${count} planet${count !== 1 ? 's' : ''}`;
    }
}

async function warmBrowserModel() {
    if (browserReady) return;
    try {
        await browserInference.init();
        browserReady = true;
    } catch (error) {
        browserReady = false;
        throw error;
    }
}

// Auto-save tycoon state every 30 seconds when active
setInterval(() => {
    if (tycoon?.active && !tycoon.paused) {
        tycoon.save();
    }
}, 30000);
