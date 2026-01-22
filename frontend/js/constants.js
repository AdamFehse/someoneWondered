/**
 * Constants for the Space Simulation Application
 * Centralizes all configuration values, colors, and magic numbers
 */

// Color definitions
export const COLORS = {
    // Background and fog
    BACKGROUND: 0x000510,
    
    // Lighting
    AMBIENT_LIGHT: 0xffffff,
    STAR_LIGHT: 0xffd700,
    
    // Default fallback colors
    DEFAULT_ORBIT_COLOR: 0x00ffff,
    
    // Starfield
    STARFIELD_COLOR: 0xffffff,
    
    // Planet color palette (no cyan or yellow as requested)
    PLANET_PALETTE: [
        0x4a90e2,  // Blue
        0xff6b6b,  // Red
        0x00cc88,  // Green
        0xffa500,  // Orange
        0xff69b4,  // Hot pink
        0x9d4edd,  // Purple
        0x50c878,  // Emerald green
        0xd4af37,  // Gold
    ]
};

// Camera settings
export const CAMERA = {
    FOV: 75,
    NEAR_PLANE: 0.1,
    FAR_PLANE: 10000,
    INITIAL_POSITION: { x: 0, y: 15, z: 15 },
    TARGET: { x: 0, y: 0, z: 0 }
};

// Lighting settings
export const LIGHTING = {
    AMBIENT_INTENSITY: 0.3,
    STAR_INTENSITY: 1.5,
    STAR_DISTANCE: 100,
    SHADOW_MAP_SIZE: 2048
};

// UI and interaction settings
export const UI = {
    CONTROL_PANEL_WIDTH: 460,
    INFO_PANEL_WIDTH: 300,
    MIN_PANEL_WIDTH: 250,
    MIN_PANEL_HEIGHT: 200,
    DRAG_THRESHOLD: 10,
    MIN_CAMERA_RADIUS: 5,
    MAX_CAMERA_RADIUS: 200,
    ZOOM_SPEED: 1.1,
    CAMERA_PHI_MIN: 0.1,
    CAMERA_PHI_MAX: Math.PI - 0.1,
    RESIZE_HANDLE_SIZE: 30
};

// Animation and timing settings
export const ANIMATION = {
    FRAME_RATE: 30,
    TIMER_INTERVAL: 100, // ms
    ROTATION_SPEED: Math.PI / 100, // radians per frame
    HIGHLIGHT_ROTATION_X_FACTOR: 0.3,
    HIGHLIGHT_ROTATION_Y_FACTOR: 1.0,
    HIGHLIGHT_ROTATION_Z_FACTOR: 0.5,
    PULSE_MIN: 0.6,
    PULSE_MAX: 1.0,
    PULSE_FREQUENCY: 2 // oscillations per second
};

// Visual effects settings
export const VISUAL_EFFECTS = {
    HIGHLIGHT_OPACITY: 0.8,
    HIGHLIGHT_EMISSIVE_INTENSITY: 0.8,
    TRAIL_OPACITY: 0.9,
    TRAIL_BLENDING: 'AdditiveBlending',
    ORBIT_PATH_OPACITY: 0.45,
    ORBIT_PATH_LINEWIDTH: 1,
    TRAIL_FADE_START: 0.3,
    TRAIL_FADE_END: 0.7,
    TORUS_RADIUS_MULTIPLIER: 0.3,
    STARFIELD_PARTICLE_SIZE: 0.7,
    STARFIELD_COUNT: 5000,
    STARFIELD_RADIUS: 500
};

// Geometry settings
export const GEOMETRY = {
    STAR_SPHERE_SEGMENTS: 32,
    PLANET_SPHERE_SEGMENTS: 16,
    STAR_RADIUS_FACTOR: 0.3,
    PLANET_MIN_RADIUS: 0.15,
    PLANET_RADIUS_FACTOR: 0.5,
    TORUS_TUBULAR_SEGMENTS: 16,
    TORUS_RADIAL_SEGMENTS: 32,
    ORBIT_PATH_SEGMENTS: 100,
    TRAIL_MAX_POSITIONS: 500,
    STARFIELD_STAR_SEGMENTS: 32
};

// Material properties
export const MATERIAL = {
    STAR_EMISSIVE_INTENSITY: 0.5,
    STAR_SHININESS: 100,
    PLANET_EMISSIVE_INTENSITY: 0.2,
    PLANET_SHININESS: 50
};

// Physics and simulation settings
export const PHYSICS = {
    SIMULATION_DT_DEFAULT: 0.01,
    CENTRAL_MASS_MIN: 0.01,
    MASS_RATIO: 0.2,
    SIMULATION_TIMESTEPS: 1000,
    TARGET_VIEW_DISTANCE: 10,
    SAMPLE_RATE_DIVISOR: 50,
    CAMERA_ROTATION_DELTA: 0.005,
    COLOR_BRIGHTENING_FACTOR: 1.5
};

// Visual interaction settings
export const INTERACTION = {
    CAMERA_ROTATION_DELTA: 0.005
};

// Color effects settings
export const COLOR_EFFECTS = {
    BRIGHTENING_FACTOR: 1.5
};

// API defaults
export const API_DEFAULTS = {
    CENTRAL_MASS: 0.5,
    NUM_BODIES: 3,
    TEMPERATURE: 0.8,
    TOP_K: 50,
    SIMULATION_TIMESTEPS: 1000,
    SIMULATION_DT: 0.01
};

// Browser inference settings
export const INFERENCE = {
    VOCAB_SIZE: 256,
    MAX_SEQ_LEN: 64,
    NUM_BINS: 253,
    OFFSET: 3
};

export const MODEL = {
    ONNX_MODEL_URL: "./assets/planet_model.onnx",
    STATS_URL: "./assets/normalization_stats.json",
    WASM_BASE_URL: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/"
};
