/**
 * Main Application Logic
 */
import { UI, PHYSICS, ANIMATION, API_DEFAULTS } from './constants.js';
import { SpaceSimulationAPI } from './api.js';
import { SpaceVisualization } from './visualization.js';
import { BrowserInference } from './browser_inference.js';
import { initializeTheme } from './theme.js';

// Global state
let visualization = null;
let api = null;
let browserInference = null;
let browserReady = false;

// Initialize on page load
document.addEventListener("DOMContentLoaded", async () => {
  // Initialize theme system (must be first for consistent styling)
  initializeTheme();

  // Initialize API (auto-detects environment: localhost for dev, Render for production)
  api = new SpaceSimulationAPI();
  browserInference = new BrowserInference();

  // Initialize visualization (now respects theme)
  visualization = new SpaceVisualization("canvas-container");

  // Setup UI event listeners
  setupEventListeners();

  updateInferenceInfo();

  // Load initial system
  await generateSystem();
});


function setupEventListeners() {
  // Generate button
  document
    .getElementById("generate-btn")
    .addEventListener("click", async () => {
      await generateSystem();
    });

  const inferenceSelect = document.getElementById("inference-mode");
  if (inferenceSelect) {
    inferenceSelect.addEventListener("change", () => {
      updateInferenceInfo();
    });
  }

  // Pause button
  document.getElementById("pause-btn").addEventListener("click", () => {
    if (visualization.isPlaying) {
      visualization.pause();
      document.getElementById("pause-btn").textContent = "Resume";
    } else {
      visualization.play();
      document.getElementById("pause-btn").textContent = "Pause";
    }
  });

  // Panel resize handle
  setupPanelResize("control-panel");
}

function setupPanelResize(panelId) {
  const panel = document.getElementById(panelId);
  if (!panel) return;

  const resizeHandle = panel.querySelector(".resize-handle");
  if (!resizeHandle) return;

  // Set initial width if not already set
  if (!panel.style.width) {
    const maxWidth = Math.max(240, window.innerWidth - 24);
    const defaultWidth = panelId === "control-panel" ? UI.CONTROL_PANEL_WIDTH : UI.INFO_PANEL_WIDTH;
    const isNarrow = window.innerWidth < 768;
    const baseWidth = isNarrow ? Math.min(320, maxWidth) : defaultWidth;
    panel.style.width = Math.min(baseWidth, maxWidth) + "px";
  }

  let isResizing = false;
  let startX, startY, startWidth, startHeight;

  const handleMouseDown = (e) => {
    if (e.button !== 0) return; // Only left mouse button
    isResizing = true;
    startX = e.clientX;
    startY = e.clientY;
    startWidth = panel.offsetWidth;
    startHeight = panel.offsetHeight;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "nwse-resize";
  };

  const handleMouseMove = (e) => {
    if (!isResizing) return;

    const deltaX = e.clientX - startX;
    const deltaY = e.clientY - startY;

    const newWidth = Math.max(UI.MIN_PANEL_WIDTH, startWidth + deltaX);
    const newHeight = Math.max(UI.MIN_PANEL_HEIGHT, startHeight + deltaY);

    panel.style.width = newWidth + "px";
    if (panelId === "info-panel") {
      panel.style.maxHeight = newHeight + "px";
    }
  };

  const handleMouseUp = () => {
    if (isResizing) {
      isResizing = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    }
  };

  resizeHandle.addEventListener("mousedown", handleMouseDown);
  document.addEventListener("mousemove", handleMouseMove);
  document.addEventListener("mouseup", handleMouseUp);

  window.addEventListener("resize", () => {
    if (window.innerWidth >= 768) {
      return;
    }
    const maxWidth = Math.max(240, window.innerWidth - 24);
    if (panel.offsetWidth > maxWidth) {
      panel.style.width = maxWidth + "px";
    }
  });
}

async function generateSystem() {
  const generateBtn = document.getElementById("generate-btn");
  const originalText = generateBtn.textContent;
  let progressLabel = "";

  // Disable button and show loading state
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";

  // Start elapsed time counter
  const startTime = Date.now();
  const timerInterval = setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
    const suffix = progressLabel ? ` ${progressLabel}` : "";
    generateBtn.textContent = `Generating...${suffix} ${elapsedSeconds}s`;
  }, 1000);

  try {
    // Get parameters from UI
    const centralMass = parseFloat(document.getElementById("central-mass").value);
    const numBodies = parseInt(document.getElementById("num-bodies").value);
    const temperature = parseFloat(document.getElementById("temperature").value);

    const safeCentralMass = Math.max(centralMass, PHYSICS.CENTRAL_MASS_MIN);
    const simulationDt = PHYSICS.SIMULATION_DT_DEFAULT * (PHYSICS.MASS_RATIO / safeCentralMass);

    const inferenceMode = getInferenceMode();
    let systemData;

    if (inferenceMode === "browser") {
      try {
        await warmBrowserModel();
        const streamResults = getStreamResults();
        systemData = await browserInference.generateSystem(
          {
            central_mass: centralMass,
            num_bodies: numBodies,
            temperature: temperature,
            top_k: API_DEFAULTS.TOP_K,
            simulation_timesteps: PHYSICS.SIMULATION_TIMESTEPS,
            simulation_dt: simulationDt
          },
          streamResults
            ? {
                onProgress: async (partialSystem, progress) => {
                  progressLabel = `${progress.completedPlanets}/${progress.totalPlanets}`;
                  visualization.loadSystem(partialSystem);
                  updateInfoPanel(partialSystem);
                }
              }
            : {}
        );
      } catch (error) {
        const message = error && error.message ? error.message : "Browser inference failed";
        console.error("Browser inference failed:", error);
        setBrowserStatus("error", message);
        return;
      }
    } else {
      systemData = await api.generateSystem({
        central_mass: centralMass,
        num_bodies: numBodies,
        temperature: temperature,
        simulation_timesteps: PHYSICS.SIMULATION_TIMESTEPS,
        simulation_dt: simulationDt
      });
    }

    if (!systemData) {
      return;
    }

    // Load into visualization (includes orbital_elements)
    visualization.loadSystem(systemData);

    // Update pause button state (loadSystem sets isPlaying = true)
    document.getElementById("pause-btn").textContent = "Pause";

    // Update info panel
    updateInfoPanel(systemData);
  } finally {
    // Clear timer
    clearInterval(timerInterval);

    // Re-enable button
    generateBtn.disabled = false;
    generateBtn.textContent = originalText;
  }
}

function updateInfoPanel(systemData) {
  document.getElementById("info-mass").textContent =
    systemData.central_mass.toFixed(2) + " M☉";
  document.getElementById("info-planets").textContent =
    Math.max(0, systemData.bodies.length - 1);
  document.getElementById("info-frame").textContent = "-";
  updateInferenceInfo();
}

// Update frame counter periodically
setInterval(() => {
  if (visualization && visualization.trajectories) {
    const progress = (
      (visualization.currentFrame / visualization.trajectories.length) *
      100
    ).toFixed(1);
    document.getElementById("info-frame").textContent =
      `${visualization.currentFrame} / ${visualization.trajectories.length} (${progress}%)`;
  }
}, ANIMATION.TIMER_INTERVAL);

function getInferenceMode() {
  const select = document.getElementById("inference-mode");
  return select ? select.value : "server";
}

function setInferenceMode(mode) {
  const select = document.getElementById("inference-mode");
  if (select) {
    select.value = mode;
  }
  updateInferenceInfo();
}

function updateInferenceInfo() {
  const mode = getInferenceMode();
  const infoEl = document.getElementById("info-inference");
  const renderNote = document.getElementById("render-note");
  if (infoEl) {
    infoEl.textContent = mode === "browser" ? "Browser (ONNX)" : "Server (Render)";
  }
  if (renderNote) {
    renderNote.classList.toggle("visible", mode === "server");
  }
}

function getStreamResults() {
  const checkbox = document.getElementById("stream-results");
  return checkbox ? checkbox.checked : false;
}

function setBrowserStatus(state, message) {
  const statusEl = document.getElementById("browser-model-status");
  if (!statusEl) return;
  statusEl.classList.remove("ready", "loading", "error");
  if (state) {
    statusEl.classList.add(state);
  }
  statusEl.textContent = message;
  statusEl.title = message;
}

async function warmBrowserModel() {
  if (browserReady) {
    return;
  }
  setBrowserStatus("loading", "Loading model...");
  try {
    await browserInference.init();
    browserReady = true;
    setBrowserStatus("ready", "Ready");
  } catch (error) {
    browserReady = false;
    const message = error && error.message ? error.message : "Model unavailable";
    console.error("Model load failed:", error);
    setBrowserStatus("error", message);
    throw error;
  }
}
