/**
 * API Helper for Backend Communication
 */
import { API_DEFAULTS } from './constants.js';

export class SpaceSimulationAPI {
    constructor(baseUrl = null) {
        // Auto-detect environment: use Render backend in production, localhost in development
        if (baseUrl) {
            this.baseUrl = baseUrl;
        } else if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            // Local development
            this.baseUrl = 'http://127.0.0.1:8000';
        } else {
            // Production (GitHub Pages) - use Render backend
            this.baseUrl = 'https://someonewondered.onrender.com';
        }
    }

    /**
     * Generate a stellar system
     */
    async generateSystem(params = {}) {
        const defaultParams = {
            central_mass: API_DEFAULTS.CENTRAL_MASS,
            num_bodies: API_DEFAULTS.NUM_BODIES,
            temperature: API_DEFAULTS.TEMPERATURE,
            top_k: API_DEFAULTS.TOP_K,
            simulation_timesteps: API_DEFAULTS.SIMULATION_TIMESTEPS,
            simulation_dt: API_DEFAULTS.SIMULATION_DT
        };

        const requestData = { ...defaultParams, ...params };

        const response = await fetch(`${this.baseUrl}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        return await response.json();
    }
}
