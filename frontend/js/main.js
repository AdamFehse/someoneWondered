import { PHYSICS, API_DEFAULTS } from './constants.js';
import { SpaceSimulationAPI } from './api.js';
import { SpaceVisualization } from './visualization.js';
import { BrowserInference } from './browser_inference.js';

let visualization = null;
let api = null;
let browserInference = null;
let browserReady = false;

const vibes = [
    'chaotic orbits ahead', 'a quiet corner', 'tidal love', 'gravity wins',
    'cosmic ballet', 'solar jazz', 'dusty neighbors', 'resonant hearts',
    'wandering giants', 'tiny & fierce', 'eccentric crew', 'just formed',
];

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

    setupEventListeners();

    await generateSystem();
});

function setupEventListeners() {
    document.getElementById("generate-btn").addEventListener("click", async () => {
        await generateSystem();
    });

    document.getElementById("pause-btn").addEventListener("click", () => {
        if (visualization.isPlaying) {
            visualization.pause();
            setPauseIcon(false);
        } else {
            visualization.play();
            setPauseIcon(true);
        }
    });

    const speedSlider = document.getElementById("speed-slider");
    const speedLabel = document.getElementById("speed-label");
    if (speedSlider) {
        speedSlider.addEventListener("input", () => {
            const m = parseFloat(speedSlider.value);
            visualization.setPlaybackSpeed(m);
            speedLabel.textContent = `${m}×`;
        });
    }
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
        const vibe = vibes[Math.floor(Math.random() * vibes.length)];
        toast(`🪐 ${Math.max(0, systemData.bodies.length - 1)} planets — ${vibe}`);
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
    document.getElementById("planet-count").textContent = `🪐 ${count} planet${count !== 1 ? 's' : ''}`;
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
