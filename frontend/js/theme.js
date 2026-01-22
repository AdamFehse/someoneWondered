/**
 * Theme Management System
 * Single source of truth for all theming across the application
 * Manages light/dark mode, localStorage persistence, and CSS variable updates
 */

export const THEME_CONFIG = {
    dark: {
        'bg-primary': '#000510',
        'bg-secondary': 'rgba(0, 5, 16, 0.2)',
        'text-primary': '#e0e0e0',
        'text-secondary': '#9090a0',
        'border-accent': 'rgba(0, 170, 255, 0.2)',
        'border-accent-hover': 'rgba(0, 170, 255, 0.8)',
        'accent-color': '#00aaff',
        'accent-light': '#0088ff',
        'gold-primary': '#ffd700',
        'gold-secondary': '#ffed4e',
        'error-primary': '#ff6b6b',
        'error-border': 'rgba(255, 107, 107, 0.2)',
        'button-bg': 'linear-gradient(135deg, #00aaff, #0088ff)',
        'button-shadow': 'rgba(0, 170, 255, 0.3)',
        'secondary-bg': 'linear-gradient(135deg, #666, #888)',
        'secondary-shadow': 'rgba(100, 100, 100, 0.3)',
        // Three.js colors (hex format)
        'canvas-bg': 0x000510,
        'canvas-accent': 0x00aaff,
        'canvas-accent-light': 0x0088ff,
    },
    light: {
        'bg-primary': '#ffffff',
        'bg-secondary': 'rgba(255, 255, 255, 0.2)',
        'text-primary': '#1a1a1a',
        'text-secondary': '#666666',
        'border-accent': 'rgba(0, 170, 255, 0.15)',
        'border-accent-hover': 'rgba(0, 170, 255, 0.6)',
        'accent-color': '#0066cc',
        'accent-light': '#0055aa',
        'gold-primary': '#e6b800',
        'gold-secondary': '#ffcc00',
        'error-primary': '#cc0000',
        'error-border': 'rgba(255, 107, 107, 0.15)',
        'button-bg': 'linear-gradient(135deg, #0066cc, #0055aa)',
        'button-shadow': 'rgba(0, 102, 204, 0.2)',
        'secondary-bg': 'linear-gradient(135deg, #cccccc, #aaaaaa)',
        'secondary-shadow': 'rgba(100, 100, 100, 0.2)',
        // Three.js colors (hex format)
        'canvas-bg': 0xf5f7fa,
        'canvas-accent': 0x0066cc,
        'canvas-accent-light': 0x0055aa,
    }
};

const STORAGE_KEY = 'theme-preference';
const THEME_CLASS = 'theme-dark';

/**
 * Get the current theme from localStorage or system preference
 */
export function getCurrentTheme() {
    // Check localStorage first
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
        return stored === 'dark' ? 'dark' : 'light';
    }

    // Fall back to system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
    }

    return 'light';
}

/**
 * Apply theme to document and CSS variables
 */
export function applyTheme(theme) {
    const config = THEME_CONFIG[theme];
    const root = document.documentElement;

    // Apply CSS variables
    Object.entries(config).forEach(([key, value]) => {
        // Skip Three.js colors in CSS variables
        if (!key.startsWith('canvas-')) {
            root.style.setProperty(`--${key}`, value);
        }
    });

    // Update document class for theme-specific styling
    if (theme === 'dark') {
        document.documentElement.classList.add(THEME_CLASS);
    } else {
        document.documentElement.classList.remove(THEME_CLASS);
    }

    // Store preference
    localStorage.setItem(STORAGE_KEY, theme);

    // Dispatch custom event for other modules
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

/**
 * Toggle between light and dark mode
 */
export function toggleTheme() {
    const current = getCurrentTheme();
    const newTheme = current === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
    return newTheme;
}

/**
 * Get CSS variable value by name
 */
export function getCSSVariable(variableName) {
    return getComputedStyle(document.documentElement).getPropertyValue(`--${variableName}`).trim();
}

/**
 * Get Three.js color value for current theme
 */
export function getCanvasColor(colorType) {
    const theme = getCurrentTheme();
    const key = `canvas-${colorType}`;
    return THEME_CONFIG[theme][key];
}

/**
 * Initialize theme system on page load
 */
export function initializeTheme() {
    const theme = getCurrentTheme();
    applyTheme(theme);

    // Listen for system theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            // Only auto-apply if user hasn't manually set a preference
            if (!localStorage.getItem(STORAGE_KEY)) {
                applyTheme(e.matches ? 'dark' : 'light');
            }
        });
    }
}

/**
 * Create theme toggle button HTML
 */
export function createThemeToggleButton() {
    const button = document.createElement('button');
    button.id = 'theme-toggle-btn';
    button.className = 'theme-toggle';
    button.setAttribute('aria-label', 'Toggle theme');
    button.innerHTML = getCurrentTheme() === 'dark' ? '☀️' : '🌙';

    button.addEventListener('click', () => {
        const newTheme = toggleTheme();
        button.innerHTML = newTheme === 'dark' ? '☀️' : '🌙';
    });

    return button;
}
