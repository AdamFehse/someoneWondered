/**
 * Main Application Logic
 */

// Global state
let visualization = null;
let api = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", async () => {
  // Initialize API (auto-detects environment: localhost for dev, Render for production)
  api = new SpaceSimulationAPI();

  // Initialize visualization
  visualization = new SpaceVisualization("canvas-container");

  // Setup UI event listeners
  setupEventListeners();

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
}

async function generateSystem() {
  const generateBtn = document.getElementById("generate-btn");
  const originalText = generateBtn.textContent;

  // Disable button and show loading state
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";

  // Start elapsed time counter
  const startTime = Date.now();
  const timerInterval = setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
    generateBtn.textContent = `Generating... ${elapsedSeconds}s`;
  }, 1000);

  try {
    // Get parameters from UI
    const centralMass = parseFloat(document.getElementById("central-mass").value);
    const numBodies = parseInt(document.getElementById("num-bodies").value);
    const temperature = parseFloat(document.getElementById("temperature").value);

    const safeCentralMass = Math.max(centralMass, 0.01);
    const simulationDt = 0.01 * (0.2 / safeCentralMass);

    // Generate system
    const systemData = await api.generateSystem({
      central_mass: centralMass,
      num_bodies: numBodies,
      temperature: temperature,
      simulation_timesteps: 1000,
      simulation_dt: simulationDt,
    });

    // Load into visualization
    visualization.loadSystem(systemData);

    if (systemData.orbital_elements) {
      console.table(systemData.orbital_elements);
    }

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
}, 100);
