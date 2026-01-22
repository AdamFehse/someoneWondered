/**
 * Three.js Visualization for Stellar Systems
 *
 * Renders N-body system with orbital trails, bloom effects, and starfield
 */

class SpaceVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000510);
        this.scene.fog = new THREE.Fog(0x000510, 100, 500);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.width / this.height,
            0.1,
            10000
        );
        this.camera.position.set(0, 15, 15);
        this.camera.lookAt(0, 0, 0);

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
        this.frameRate = 30;

        // Camera controls
        this.setupOrbitControls();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Animation
        this.animate();
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
        this.scene.add(ambientLight);

        // Point light from central star
        this.starLight = new THREE.PointLight(0xffd700, 1.5, 100);
        this.starLight.position.set(0, 0, 0);
        this.starLight.castShadow = true;
        this.starLight.shadow.mapSize.width = 2048;
        this.starLight.shadow.mapSize.height = 2048;
        this.scene.add(this.starLight);
    }

    setupOrbitControls() {
        // Camera controls state
        this.cameraControls = {
            spherical: {
                radius: 30,
                theta: 0,        // Azimuthal angle
                phi: Math.PI / 3 // Polar angle
            },
            isDragging: false,
            previousMousePosition: { x: 0, y: 0 },
            minRadius: 5,
            maxRadius: 200
        };

        // Mouse events
        this.renderer.domElement.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.renderer.domElement.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.renderer.domElement.addEventListener('wheel', (e) => this.onMouseWheel(e), { passive: false });

        this.updateCameraPosition();
    }

    onMouseDown(e) {
        if (e.button === 0) { // Left click
            this.cameraControls.isDragging = true;
            this.cameraControls.previousMousePosition = { x: e.clientX, y: e.clientY };
        }
    }

    onMouseMove(e) {
        if (!this.cameraControls.isDragging) return;

        const deltaX = e.clientX - this.cameraControls.previousMousePosition.x;
        const deltaY = e.clientY - this.cameraControls.previousMousePosition.y;

        this.cameraControls.spherical.theta -= deltaX * 0.005;
        this.cameraControls.spherical.phi -= deltaY * 0.005;

        // Clamp phi to avoid flipping
        this.cameraControls.spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.cameraControls.spherical.phi));

        this.cameraControls.previousMousePosition = { x: e.clientX, y: e.clientY };
        this.updateCameraPosition();
    }

    onMouseUp(e) {
        this.cameraControls.isDragging = false;
    }

    onMouseWheel(e) {
        e.preventDefault();
        const zoomSpeed = 1.1;
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
        const starPositions = new Float32Array(5000 * 3);

        // Generate random star positions on sphere
        for (let i = 0; i < 5000; i++) {
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI;
            const r = 500;

            starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            starPositions[i * 3 + 2] = r * Math.cos(phi);
        }

        starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));

        const starMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.7,
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
         *   bodies: [{mass, position, velocity}]
         * }
         */

        // Clear previous bodies
        this.clearBodies();

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

        console.log(`Loaded system with ${numBodies} bodies, ${this.trajectories.length} frames`);
    }

    createStar(mass) {
        // Star geometry - scale by mass
        const radius = Math.pow(mass, 1/3) * 0.3;
        const geometry = new THREE.SphereGeometry(radius, 32, 32);

        // Material with emission
        const material = new THREE.MeshPhongMaterial({
            color: 0xffd700,
            emissive: 0xffd700,
            emissiveIntensity: 0.5,
            shininess: 100
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
        // Body geometry - scale by mass (logarithmic)
        const radius = Math.pow(mass, 1/3) * 0.1;
        const geometry = new THREE.SphereGeometry(radius, 16, 16);

        // Color based on mass
        let color;
        if (mass < 0.01) {
            color = 0x00aaff;  // Blue for small bodies
        } else if (mass < 0.1) {
            color = 0x00cc88;  // Green for medium
        } else {
            color = 0xff6b6b;  // Red for large
        }

        const material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.2,
            shininess: 50
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        // Create trail
        const trail = this.createTrail();

        this.scene.add(mesh);
        this.scene.add(trail.lineSegments);

        this.bodies.push({
            mesh: mesh,
            trail: trail,
            mass: mass,
            type: 'planet',
            index: index
        });
    }

    createTrail() {
        /**
         * Create orbital trail with gradient fade effect
         */
        const geometry = new THREE.BufferGeometry();
        const maxPositions = 500;

        const positions = new Float32Array(maxPositions * 3);
        const colors = new Float32Array(maxPositions * 3);

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.LineBasicMaterial({
            vertexColors: true,
            linewidth: 2,
            fog: false,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending
        });

        const lineSegments = new THREE.LineSegments(geometry, material);
        lineSegments.castShadow = false;
        lineSegments.receiveShadow = false;

        return {
            lineSegments: lineSegments,
            positions: [],
            maxPositions: maxPositions
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

        const n = trail.positions.length;
        for (let i = 0; i < n; i++) {
            const pos = trail.positions[i];
            const idx = i * 3;

            positions[idx] = pos[0];
            positions[idx + 1] = pos[1];
            positions[idx + 2] = pos[2];

            // Brighter glow with a soft fade
            const alpha = i / n;
            colors[idx] = 0.2 + 0.8 * alpha;      // R
            colors[idx + 1] = 0.4 + 0.9 * alpha;  // G
            colors[idx + 2] = 0.8 + 1.0 * alpha;  // B
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

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    calculateScaleFactor(trajectories) {
        /**
         * Find maximum distance from center and return scale factor
         * to fit system in range of 5-15 units from center
         */
        let maxDistance = 0;

        // Sample frames to find max distance efficiently
        const sampleRate = Math.max(1, Math.floor(trajectories.length / 50));

        for (let t = 0; t < trajectories.length; t += sampleRate) {
            const frame = trajectories[t];
            for (let i = 1; i < frame.length; i++) {  // Skip star at index 0
                const pos = frame[i];
                const distance = Math.sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
                maxDistance = Math.max(maxDistance, distance);
            }
        }

        // Target viewing range: keep max distance around 10 units
        const targetDistance = 10;
        const scaleFactor = maxDistance > 0 ? targetDistance / maxDistance : 1;

        console.log(`Scaling trajectories: max distance ${maxDistance.toFixed(2)} AU → scale factor ${scaleFactor.toFixed(4)}`);
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
