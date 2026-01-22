/**
 * Three.js Visualization for Stellar Systems
 *
 * Renders N-body system with orbital trails, bloom effects, and starfield
 */
import { getThemeColors, CAMERA, LIGHTING, UI, ANIMATION, VISUAL_EFFECTS, GEOMETRY, MATERIAL, PHYSICS, INTERACTION, COLOR_EFFECTS } from './constants.js';

/**
 * Validates and sanitizes orbital element data
 */
class OrbitalValidator {
    static normalizeFromAPI(elem) {
        if (!elem) return null;

        // Map API field names to canonical names
        return {
            semi_major_axis: elem.semi_major_axis,
            eccentricity: elem.eccentricity,
            inclination: elem.inclination,
            longitude_ascending_node: elem.long_ascending_node || elem.longitude_ascending_node || 0,
            argument_periapsis: elem.arg_periapsis || elem.argument_periapsis || 0
        };
    }

    static isValid(elem) {
        if (!elem) return false;

        return Number.isFinite(elem.semi_major_axis) && elem.semi_major_axis > 0 &&
               Number.isFinite(elem.eccentricity) &&
               Number.isFinite(elem.inclination) &&
               Number.isFinite(elem.longitude_ascending_node) &&
               Number.isFinite(elem.argument_periapsis);
    }

    static sanitize(elem) {
        return OrbitalValidator.isValid(elem) ? elem : null;
    }

    static sanitizeArray(elements) {
        return elements.map((elem, idx) => {
            const normalized = OrbitalValidator.normalizeFromAPI(elem);
            const sanitized = OrbitalValidator.sanitize(normalized);
            return sanitized;
        });
    }
}

class SpaceVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        // Get theme-aware colors
        this.colors = getThemeColors();

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.colors.BACKGROUND);
        this.scene.fog = new THREE.Fog(this.colors.BACKGROUND, LIGHTING.STAR_DISTANCE, PHYSICS.TARGET_VIEW_DISTANCE * 50);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            CAMERA.FOV,
            this.width / this.height,
            CAMERA.NEAR_PLANE,
            CAMERA.FAR_PLANE
        );
        this.camera.position.set(CAMERA.INITIAL_POSITION.x, CAMERA.INITIAL_POSITION.y, CAMERA.INITIAL_POSITION.z);
        this.camera.lookAt(CAMERA.TARGET.x, CAMERA.TARGET.y, CAMERA.TARGET.z);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        this.setupLighting();

        // Starfield background
        this.createStarfield();

        // Object tracking
        this.bodies = [];
        this.trails = [];
        this.trajectories = null;
        this.currentFrame = 0;
        this.isPlaying = true;
        this.frameRate = ANIMATION.FRAME_RATE;

        // Camera controls
        this.setupOrbitControls();

        // Selection system
        this.setupSelection();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Listen for theme changes
        window.addEventListener('themechange', () => this.updateTheme());

        // Animation
        this.animate();
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(this.colors.AMBIENT_LIGHT, LIGHTING.AMBIENT_INTENSITY);
        this.scene.add(ambientLight);

        // Point light from central star
        this.starLight = new THREE.PointLight(this.colors.STAR_LIGHT, LIGHTING.STAR_INTENSITY, LIGHTING.STAR_DISTANCE);
        this.starLight.position.set(CAMERA.TARGET.x, CAMERA.TARGET.y, CAMERA.TARGET.z);
        this.starLight.castShadow = true;
        this.starLight.shadow.mapSize.width = LIGHTING.SHADOW_MAP_SIZE;
        this.starLight.shadow.mapSize.height = LIGHTING.SHADOW_MAP_SIZE;
        this.scene.add(this.starLight);
    }

    setupOrbitControls() {
        // Camera controls state
        this.cameraControls = {
            spherical: {
                radius: UI.MAX_CAMERA_RADIUS / 6,  // 30
                theta: 0,        // Azimuthal angle
                phi: Math.PI / 3 // Polar angle
            },
            isDragging: false,
            previousMousePosition: { x: 0, y: 0 },
            dragThreshold: UI.DRAG_THRESHOLD,
            dragDistance: 0,
            minRadius: UI.MIN_CAMERA_RADIUS,
            maxRadius: UI.MAX_CAMERA_RADIUS,
            dragStartedOverBody: false
        };

        // Mouse events
        this.renderer.domElement.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.renderer.domElement.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.renderer.domElement.addEventListener('wheel', (e) => this.onMouseWheel(e), { passive: false });

        this.updateCameraPosition();
    }

    setupSelection() {
        // Selection state
        this.selection = {
            bodyIndex: null,
            highlightRing: null,
            orbitPath: null,
            originalTrailColors: null
        };

        // Raycasting for click detection
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        // Setup selection event handlers
        this.setupSelectionHandlers();
    }

    setupSelectionHandlers() {
        // Click detection will be handled in onMouseUp to distinguish from drag
    }

    onBodyClick(event) {
        // Calculate mouse position in normalized device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);

        // Get intersects with body meshes only
        const bodyMeshes = this.bodies.map(b => b.mesh);
        const intersects = this.raycaster.intersectObjects(bodyMeshes);

        if (intersects.length > 0) {
            // Find which body was clicked
            const clickedMesh = intersects[0].object;
            const bodyIndex = this.bodies.findIndex(b => b.mesh === clickedMesh);

            if (bodyIndex !== -1) {
                if (this.selection.bodyIndex === bodyIndex) {
                    // Deselect if clicking same body
                    this.deselectBody();
                } else {
                    // Select new body
                    this.selectBody(bodyIndex);
                }
            }
        } else {
            // Clicked empty space - deselect
            this.deselectBody();
        }
    }

    selectBody(bodyIndex) {
        // Deselect previous if any
        if (this.selection.bodyIndex !== null) {
            this.deselectBody();
        }

        this.selection.bodyIndex = bodyIndex;
        const body = this.bodies[bodyIndex];

        // Create highlight ring
        this.selection.highlightRing = this.createHighlightRing(body);
        this.scene.add(this.selection.highlightRing);

        // Create orbital path if orbital elements available
        if (this.systemData && this.systemData.orbital_elements && bodyIndex > 0) {
            this.selection.orbitPath = this.createOrbitPath(bodyIndex);
            if (this.selection.orbitPath) {
                this.scene.add(this.selection.orbitPath);
            }
        }

        // Modify trail colors for neon effect
        if (body.trail) {
            const colors = body.trail.lineSegments.geometry.attributes.color.array;
            this.selection.originalTrailColors = new Float32Array(colors);
            this.applyNeonTrailEffect(body);
        }

        // Update UI with selection
        this.updateSelectionUI(bodyIndex);
    }

    createHighlightRing(body) {
        // Get body radius and create torus at 2x radius
        const bodyMesh = body.mesh;
        const bodyRadius = (bodyMesh.geometry.parameters.radius || GEOMETRY.PLANET_RADIUS_FACTOR) * 2;
        const torusRadius = bodyRadius * 2;

        // Create torus geometry
        const torusGeometry = new THREE.TorusGeometry(torusRadius, bodyRadius * VISUAL_EFFECTS.TORUS_RADIUS_MULTIPLIER, GEOMETRY.TORUS_TUBULAR_SEGMENTS, GEOMETRY.TORUS_RADIAL_SEGMENTS);

        // Use planet's color for highlight ring
        const ringColor = body.color || this.colors.DEFAULT_ORBIT_COLOR;
        const torusMaterial = new THREE.MeshPhongMaterial({
            color: ringColor,
            emissive: ringColor,
            emissiveIntensity: VISUAL_EFFECTS.HIGHLIGHT_EMISSIVE_INTENSITY,
            transparent: true,
            opacity: VISUAL_EFFECTS.HIGHLIGHT_OPACITY,
            wireframe: false
        });

        const torusMesh = new THREE.Mesh(torusGeometry, torusMaterial);
        torusMesh.userData = {
            creationTime: Date.now(),
            bodyIndex: this.selection.bodyIndex
        };
        // Make highlight ring non-interactive (don't interfere with raycasting)
        torusMesh.raycast = () => { };

        return torusMesh;
    }

    deselectBody() {
        if (this.selection.bodyIndex === null) return;

        // Clean up visual effects
        if (this.selection.highlightRing) {
            this.scene.remove(this.selection.highlightRing);
            this.selection.highlightRing = null;
        }

        if (this.selection.orbitPath) {
            this.scene.remove(this.selection.orbitPath);
            this.selection.orbitPath = null;
        }

        // Restore trail colors if they were changed
        if (this.selection.originalTrailColors) {
            const body = this.bodies[this.selection.bodyIndex];
            if (body && body.trail) {
                const colors = body.trail.lineSegments.geometry.attributes.color.array;
                const original = this.selection.originalTrailColors;
                for (let i = 0; i < original.length; i++) {
                    colors[i] = original[i];
                }
                body.trail.lineSegments.geometry.attributes.color.needsUpdate = true;
            }
            this.selection.originalTrailColors = null;
        }

        this.selection.bodyIndex = null;

        // Hide selection UI
        this.hideSelectionUI();
    }

    updateSelectionUI(bodyIndex) {
        const body = this.bodies[bodyIndex];
        const selectionSection = document.getElementById('selection-section');

        // Show selection section with smooth animation
        selectionSection.classList.remove('hidden');
        selectionSection.classList.add('visible');

        // Update body name (star or Planet #N)
        let bodyName = 'Star';
        if (body.type === 'planet') {
            bodyName = `Planet #${body.index}`;
        }
        document.getElementById('selection-name').textContent = bodyName;

        // Update mass
        document.getElementById('selection-mass').textContent =
            body.mass.toFixed(4) + ' M☉';

        // Update orbital elements if available
        const smaEl = document.getElementById('selection-sma');
        const eccEl = document.getElementById('selection-ecc');
        const incEl = document.getElementById('selection-inc');
        const periodEl = document.getElementById('selection-period');

        if (this.systemData && this.systemData.orbital_elements && bodyIndex > 0 && this.systemData.orbital_elements[bodyIndex]) {
            const orbital = this.systemData.orbital_elements[bodyIndex];
            smaEl.textContent = orbital.semi_major_axis != null ? orbital.semi_major_axis.toFixed(3) + ' AU' : '-';
            eccEl.textContent = orbital.eccentricity != null ? orbital.eccentricity.toFixed(4) : '-';
            incEl.textContent = orbital.inclination != null ? orbital.inclination.toFixed(2) + '°' : '-';
            periodEl.textContent = orbital.orbital_period != null ? orbital.orbital_period.toFixed(3) + ' years' : '-';
        } else {
            // Star or no orbital elements available
            smaEl.textContent = '-';
            eccEl.textContent = '-';
            incEl.textContent = '-';
            periodEl.textContent = '-';
        }

        // Setup deselect button
        const deselectBtn = document.getElementById('deselect-btn');
        deselectBtn.onclick = () => this.deselectBody();
    }

    hideSelectionUI() {
        const selectionSection = document.getElementById('selection-section');
        selectionSection.classList.remove('visible');
        selectionSection.classList.add('hidden');
    }

    onMouseDown(e) {
        if (e.button === 0) { // Left click
            this.cameraControls.isDragging = true;
            this.cameraControls.previousMousePosition = { x: e.clientX, y: e.clientY };
            this.cameraControls.dragDistance = 0;

            // Check if starting drag over a body
            const rect = this.renderer.domElement.getBoundingClientRect();
            this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

            this.raycaster.setFromCamera(this.mouse, this.camera);
            const bodyMeshes = this.bodies.map(b => b.mesh);
            const intersects = this.raycaster.intersectObjects(bodyMeshes);

            this.cameraControls.dragStartedOverBody = intersects.length > 0;
        }
    }

    onMouseMove(e) {
        if (!this.cameraControls.isDragging) return;

        const deltaX = e.clientX - this.cameraControls.previousMousePosition.x;
        const deltaY = e.clientY - this.cameraControls.previousMousePosition.y;

        // Track drag distance
        this.cameraControls.dragDistance += Math.sqrt(deltaX * deltaX + deltaY * deltaY);

        // Only apply camera movement if dragged beyond threshold and didn't start over a body
        if (this.cameraControls.dragDistance > this.cameraControls.dragThreshold && !this.cameraControls.dragStartedOverBody) {
            this.cameraControls.spherical.theta -= deltaX * INTERACTION.CAMERA_ROTATION_DELTA;
            this.cameraControls.spherical.phi -= deltaY * INTERACTION.CAMERA_ROTATION_DELTA;

            // Clamp phi to avoid flipping
            this.cameraControls.spherical.phi = Math.max(UI.CAMERA_PHI_MIN, Math.min(UI.CAMERA_PHI_MAX, this.cameraControls.spherical.phi));

            this.updateCameraPosition();
        }

        this.cameraControls.previousMousePosition = { x: e.clientX, y: e.clientY };
    }

    onMouseUp(e) {
        if (e.button === 0) { // Only process left click
            if (this.cameraControls.isDragging && this.cameraControls.dragDistance < this.cameraControls.dragThreshold) {
                // This was a click, not a drag
                this.onBodyClick(e);
            }
            this.cameraControls.isDragging = false;
            this.cameraControls.dragDistance = 0;
        }
    }

    onMouseWheel(e) {
        e.preventDefault();
        const zoomSpeed = UI.ZOOM_SPEED;
        if (e.deltaY > 0) {
            this.cameraControls.spherical.radius *= zoomSpeed;
        } else {
            this.cameraControls.spherical.radius /= zoomSpeed;
        }

        this.cameraControls.spherical.radius = Math.max(
            this.cameraControls.minRadius,
            Math.min(this.cameraControls.maxRadius, this.cameraControls.spherical.radius)
        );

        this.updateCameraPosition();
    }

    updateCameraPosition() {
        const sc = this.cameraControls.spherical;
        const x = sc.radius * Math.sin(sc.phi) * Math.cos(sc.theta);
        const y = sc.radius * Math.cos(sc.phi);
        const z = sc.radius * Math.sin(sc.phi) * Math.sin(sc.theta);

        this.camera.position.set(x, y, z);
        this.camera.lookAt(0, 0, 0);
    }

    createStarfield() {
        const starGeometry = new THREE.BufferGeometry();
        const starPositions = new Float32Array(VISUAL_EFFECTS.STARFIELD_COUNT * 3);

        // Generate random star positions on sphere
        for (let i = 0; i < VISUAL_EFFECTS.STARFIELD_COUNT; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;
            const r = VISUAL_EFFECTS.STARFIELD_RADIUS;

            starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            starPositions[i * 3 + 2] = r * Math.cos(phi);
        }

        starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));

        const starMaterial = new THREE.PointsMaterial({
            color: this.colors.STARFIELD_COLOR,
            size: VISUAL_EFFECTS.STARFIELD_PARTICLE_SIZE,
            sizeAttenuation: true
        });

        this.stars = new THREE.Points(starGeometry, starMaterial);
        this.scene.add(this.stars);
    }

    loadSystem(systemData) {
        /**
         * Load stellar system for visualization
         *
         * systemData: {
         *   central_mass: float,
         *   trajectory: [timestep][body][x,y,z],
         *   bodies: [{mass, position, velocity}],
         *   orbital_elements: [...]
         * }
         */

        // Clear selection when loading new system
        this.deselectBody();

        // Clear previous bodies
        this.clearBodies();

        // Store orbital elements for selection info
        // Validate and sanitize orbital elements early
        if (systemData.orbital_elements) {
            systemData.orbital_elements = OrbitalValidator.sanitizeArray(systemData.orbital_elements);
        }
        this.systemData = systemData;

        // Calculate and apply scaling to fit trajectories in viewable range
        const scaleFactor = this.calculateScaleFactor(systemData.trajectory);
        this.trajectories = this.scaleTrajectories(systemData.trajectory, scaleFactor);

        const centralMass = systemData.central_mass;
        const numBodies = systemData.bodies.length;

        // Create central star
        this.createStar(centralMass);

        // Create orbital bodies
        for (let i = 1; i < numBodies; i++) {
            this.createBody(i, systemData.bodies[i].mass);
        }

        // Reset animation
        this.currentFrame = 0;
        this.isPlaying = true;
    }

    createStar(mass) {
        // Star geometry - scale by mass
        const radius = Math.pow(mass, 1/3) * GEOMETRY.STAR_RADIUS_FACTOR;
        const geometry = new THREE.SphereGeometry(radius, GEOMETRY.STAR_SPHERE_SEGMENTS, GEOMETRY.STAR_SPHERE_SEGMENTS);

        // Material with emission
        const material = new THREE.MeshPhongMaterial({
            color: this.colors.STAR_LIGHT,
            emissive: this.colors.STAR_LIGHT,
            emissiveIntensity: MATERIAL.STAR_EMISSIVE_INTENSITY,
            shininess: MATERIAL.STAR_SHININESS
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        // Add to scene
        this.scene.add(mesh);

        this.bodies.push({
            mesh: mesh,
            trail: null,
            mass: mass,
            type: 'star'
        });
    }

    createBody(index, mass) {
        // Body geometry - scale by mass with minimum size for visibility
        const radius = Math.max(GEOMETRY.PLANET_MIN_RADIUS, Math.pow(mass, 1/3) * GEOMETRY.PLANET_RADIUS_FACTOR);
        const geometry = new THREE.SphereGeometry(radius, GEOMETRY.PLANET_SPHERE_SEGMENTS, GEOMETRY.PLANET_SPHERE_SEGMENTS);

        // Color palette - cycle through distinct colors by planet index
        const colorPalette = this.colors.PLANET_PALETTE;
        const color = colorPalette[(index - 1) % colorPalette.length];

        const material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: MATERIAL.PLANET_EMISSIVE_INTENSITY,
            shininess: MATERIAL.PLANET_SHININESS
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        // Create trail with planet's color
        const trail = this.createTrail(color);

        this.scene.add(mesh);
        this.scene.add(trail.lineSegments);

        this.bodies.push({
            mesh: mesh,
            trail: trail,
            mass: mass,
            type: 'planet',
            index: index,
            color: color
        });
    }

    createTrail(color) {
        /**
         * Create orbital trail with gradient fade effect
         * color: hex color value (e.g., 0x00aaff)
         */
        const geometry = new THREE.BufferGeometry();
        const maxPositions = GEOMETRY.TRAIL_MAX_POSITIONS;

        const positions = new Float32Array(maxPositions * 3);
        const colors = new Float32Array(maxPositions * 3);

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Disable automatic bounding sphere computation to avoid NaN issues
        geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), PHYSICS.TARGET_VIEW_DISTANCE);

        const material = new THREE.LineBasicMaterial({
            vertexColors: true,
            linewidth: 2, // Could make this configurable too
            fog: false,
            transparent: true,
            opacity: VISUAL_EFFECTS.TRAIL_OPACITY,
            blending: THREE[VISUAL_EFFECTS.TRAIL_BLENDING]
        });

        const lineSegments = new THREE.LineSegments(geometry, material);
        lineSegments.castShadow = false;
        lineSegments.receiveShadow = false;
        lineSegments.frustumCulled = false;

        return {
            lineSegments: lineSegments,
            positions: [],
            maxPositions: maxPositions,
            color: color
        };
    }

    updateTrail(body, position) {
        const trail = body.trail;
        if (!trail) return;

        // Add position to trail
        trail.positions.push([position[0], position[1], position[2]]);

        // Keep only last N positions
        if (trail.positions.length > trail.maxPositions) {
            trail.positions.shift();
        }

        // Update geometry
        const geometry = trail.lineSegments.geometry;
        const positions = geometry.attributes.position.array;
        const colors = geometry.attributes.color.array;

        // Extract RGB from planet's color
        const planetColor = trail.color;
        const r = ((planetColor >> 16) & 255) / 255;
        const g = ((planetColor >> 8) & 255) / 255;
        const b = (planetColor & 255) / 255;

        const n = trail.positions.length;
        for (let i = 0; i < n; i++) {
            const pos = trail.positions[i];
            const idx = i * 3;

            positions[idx] = pos[0];
            positions[idx + 1] = pos[1];
            positions[idx + 2] = pos[2];

            // Fade effect: dim at start, brighten towards end
            const alpha = i / n;
            colors[idx] = r * (VISUAL_EFFECTS.TRAIL_FADE_START + VISUAL_EFFECTS.TRAIL_FADE_END * alpha);
            colors[idx + 1] = g * (VISUAL_EFFECTS.TRAIL_FADE_START + VISUAL_EFFECTS.TRAIL_FADE_END * alpha);
            colors[idx + 2] = b * (VISUAL_EFFECTS.TRAIL_FADE_START + VISUAL_EFFECTS.TRAIL_FADE_END * alpha);
        }

        geometry.attributes.position.needsUpdate = true;
        geometry.attributes.color.needsUpdate = true;
        geometry.setDrawRange(0, n - 1);
    }

    updateBodies(frameIndex) {
        if (!this.trajectories || frameIndex >= this.trajectories.length) {
            return;
        }

        const frame = this.trajectories[frameIndex];

        for (let i = 0; i < this.bodies.length; i++) {
            const body = this.bodies[i];
            const pos = frame[i];

            body.mesh.position.set(pos[0], pos[1], pos[2]);

            // Update trail
            if (body.trail) {
                this.updateTrail(body, pos);
            }
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.isPlaying && this.trajectories) {
            // Update bodies
            this.updateBodies(this.currentFrame);

            // Advance frame
            this.currentFrame++;
            if (this.currentFrame >= this.trajectories.length) {
                this.currentFrame = 0;
                this.resetTrails();
            }
        }

        // Update selection highlights (animation, position tracking, etc.)
        if (this.selection.bodyIndex !== null) {
            this.updateSelectionHighlights();
        }

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    createOrbitPath(bodyIndex) {
        if (!this.systemData || !this.systemData.orbital_elements) return null;

        const orbitalElements = this.systemData.orbital_elements[bodyIndex];
        if (!OrbitalValidator.isValid(orbitalElements)) return null;

        const body = this.bodies[bodyIndex];
        const orbitColor = body ? body.color : this.colors.DEFAULT_ORBIT_COLOR;

        const a = orbitalElements.semi_major_axis;
        const e = orbitalElements.eccentricity;
        const i = (orbitalElements.inclination * Math.PI) / 180;  // Convert to radians
        const Omega = (orbitalElements.longitude_ascending_node * Math.PI) / 180;
        const omega = (orbitalElements.argument_periapsis * Math.PI) / 180;

        // Generate elliptical orbit points
        const points = [];
        const segments = GEOMETRY.ORBIT_PATH_SEGMENTS;

        for (let j = 0; j <= segments; j++) {
            const theta = (j / segments) * Math.PI * 2;

            // Ellipse equation: r = a(1-e²)/(1+e*cos(theta))
            const r = (a * (1 - e * e)) / (1 + e * Math.cos(theta));

            // Position in orbital plane
            const x = r * Math.cos(theta);
            const y = r * Math.sin(theta);

            // Apply orbital inclination and rotation
            // This is a simplified 3D rotation - proper implementation would use rotation matrices
            const cosI = Math.cos(i);
            const sinI = Math.sin(i);
            const cosOmega = Math.cos(Omega);
            const sinOmega = Math.sin(Omega);
            const cosOmega2 = Math.cos(omega);
            const sinOmega2 = Math.sin(omega);

            // Rotation matrix multiplication (simplified)
            const px = (cosOmega * cosOmega2 - sinOmega * sinOmega2 * cosI) * x +
                       (-cosOmega * sinOmega2 - sinOmega * cosOmega2 * cosI) * y;
            const py = (sinOmega * cosOmega2 + cosOmega * sinOmega2 * cosI) * x +
                       (-sinOmega * sinOmega2 + cosOmega * cosOmega2 * cosI) * y;
            const pz = sinI * sinOmega2 * x + sinI * cosOmega2 * y;

            points.push(new THREE.Vector3(px, py, pz));
        }

        // Create line geometry
        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
        const lineMaterial = new THREE.LineBasicMaterial({
            color: orbitColor,
            opacity: VISUAL_EFFECTS.ORBIT_PATH_OPACITY,
            transparent: true,
            linewidth: VISUAL_EFFECTS.ORBIT_PATH_LINEWIDTH,
            fog: false
        });

        const line = new THREE.Line(lineGeometry, lineMaterial);
        line.userData = { bodyIndex: bodyIndex };
        // Make orbit path non-interactive (don't interfere with raycasting/clicks)
        line.raycast = () => { };

        return line;
    }

    applyNeonTrailEffect(body) {
        if (!body.trail) return;

        const colors = body.trail.lineSegments.geometry.attributes.color.array;
        const n = body.trail.positions.length;

        // Extract RGB from planet's color and brighten for neon effect
        const planetColor = body.color;
        const r = Math.min(1.0, ((planetColor >> 16) & 255) / 255 * COLOR_EFFECTS.BRIGHTENING_FACTOR);
        const g = Math.min(1.0, ((planetColor >> 8) & 255) / 255 * COLOR_EFFECTS.BRIGHTENING_FACTOR);
        const b = Math.min(1.0, (planetColor & 255) / 255 * COLOR_EFFECTS.BRIGHTENING_FACTOR);

        for (let i = 0; i < n; i++) {
            colors[i * 3] = r;
            colors[i * 3 + 1] = g;
            colors[i * 3 + 2] = b;
        }

        body.trail.lineSegments.geometry.attributes.color.needsUpdate = true;
    }

    updateSelectionHighlights() {
        if (this.selection.bodyIndex === null || !this.selection.highlightRing) return;

        const body = this.bodies[this.selection.bodyIndex];
        const ring = this.selection.highlightRing;

        // Position ring at body location
        ring.position.copy(body.mesh.position);

        // Rotate ring (0.5 RPM = ~3 degrees per frame at 60fps)
        const rotationSpeed = ANIMATION.ROTATION_SPEED; // Radians per frame
        ring.rotation.x += rotationSpeed * ANIMATION.HIGHLIGHT_ROTATION_X_FACTOR;
        ring.rotation.y += rotationSpeed * ANIMATION.HIGHLIGHT_ROTATION_Y_FACTOR;
        ring.rotation.z += rotationSpeed * ANIMATION.HIGHLIGHT_ROTATION_Z_FACTOR;

        // Pulse opacity
        const elapsedSeconds = (Date.now() - ring.userData.creationTime) / 1000;
        const pulse = ANIMATION.PULSE_MIN + (ANIMATION.PULSE_MAX - ANIMATION.PULSE_MIN) * Math.sin(elapsedSeconds * ANIMATION.PULSE_FREQUENCY);  // Oscillates between min and max
        ring.material.opacity = pulse;

        // Orbit path is already centered at star (origin), no need to reposition
    }

    calculateScaleFactor(trajectories) {
        /**
         * Find maximum distance from center and return scale factor
         * to fit system in range of 5-15 units from center
         */
        let maxDistance = 0;

        // Sample frames to find max distance efficiently
        const sampleRate = Math.max(1, Math.floor(trajectories.length / PHYSICS.SAMPLE_RATE_DIVISOR));

        for (let t = 0; t < trajectories.length; t += sampleRate) {
            const frame = trajectories[t];
            for (let i = 1; i < frame.length; i++) {  // Skip star at index 0
                const pos = frame[i];
                const distance = Math.sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
                maxDistance = Math.max(maxDistance, distance);
            }
        }

        // Target viewing range: keep max distance around target units
        const targetDistance = PHYSICS.TARGET_VIEW_DISTANCE;
        const scaleFactor = maxDistance > 0 ? targetDistance / maxDistance : 1;

        return scaleFactor;
    }

    scaleTrajectories(trajectories, scaleFactor) {
        /**
         * Apply scaling factor to all positions in trajectories
         */
        return trajectories.map(frame =>
            frame.map(pos => [pos[0] * scaleFactor, pos[1] * scaleFactor, pos[2] * scaleFactor])
        );
    }

    clearBodies() {
        for (const body of this.bodies) {
            this.scene.remove(body.mesh);
            if (body.trail) {
                this.scene.remove(body.trail.lineSegments);
            }
        }
        this.bodies = [];
        this.trails = [];
    }

    onWindowResize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(this.width, this.height);
    }

    updateTheme() {
        // Update theme-aware colors
        this.colors = getThemeColors();

        // Update scene background and fog
        this.scene.background.setHex(this.colors.BACKGROUND);
        this.scene.fog.color.setHex(this.colors.BACKGROUND);

        // Note: Other materials (planets, trails, etc.) will remain with their original colors
        // Only background and orbit lines that use DEFAULT_ORBIT_COLOR will update
        // This prevents sudden color shifts of planets when theme changes
    }

    play() {
        this.isPlaying = true;
    }

    pause() {
        this.isPlaying = false;
    }

    resetTrails() {
        for (const body of this.bodies) {
            if (!body.trail) {
                continue;
            }
            body.trail.positions = [];
            const geometry = body.trail.lineSegments.geometry;
            geometry.setDrawRange(0, 0);
            geometry.attributes.position.needsUpdate = true;
            geometry.attributes.color.needsUpdate = true;
        }
    }

    reset() {
        this.currentFrame = 0;
        this.resetTrails();

    }
}

export { SpaceVisualization };
