/**
 * API Helper for Backend Communication
 */

class SpaceSimulationAPI {
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
            central_mass: 0.5,
            num_bodies: 3,
            temperature: 0.8,
            top_k: 50,
            simulation_timesteps: 1000,
            simulation_dt: 0.01
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
