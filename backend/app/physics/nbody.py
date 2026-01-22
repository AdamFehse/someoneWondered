"""
N-Body Gravitational Physics Simulator

Implements velocity Verlet integrator for symplectic integration.
"""

import numpy as np
from typing import Tuple, Dict, List


class OrbitalMechanics:
    """Convert between orbital elements and Cartesian coordinates"""

    @staticmethod
    def orbital_to_cartesian(
        semi_major_axis: float,
        eccentricity: float,
        inclination: float,
        arg_periapsis: float,
        long_asc_node: float,
        mean_anomaly: float,
        mu: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert orbital elements to Cartesian position and velocity

        Args:
            semi_major_axis: Semi-major axis (AU)
            eccentricity: Orbital eccentricity [0, 1)
            inclination: Inclination angle (rad)
            arg_periapsis: Argument of periapsis (rad)
            long_asc_node: Longitude of ascending node (rad)
            mean_anomaly: Mean anomaly (rad)

        Returns:
            position: (3,) array in AU
            velocity: (3,) array in AU/year
        """
        # Solve Kepler's equation for eccentric anomaly (Newton's method)
        E = mean_anomaly
        for _ in range(6):
            f = E - eccentricity * np.sin(E) - mean_anomaly
            f_prime = 1 - eccentricity * np.cos(E) + 1e-8
            E = E - f / f_prime

        # Position in orbital plane
        r = semi_major_axis * (1 - eccentricity * np.cos(E))
        nu = 2 * np.arctan2(
            np.sqrt(1 + eccentricity) * np.sin(E / 2),
            np.sqrt(1 - eccentricity) * np.cos(E / 2)
        )

        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0.0

        # Velocity in orbital plane (two-body)
        h = np.sqrt(mu * semi_major_axis * (1 - eccentricity ** 2))
        v_r = (mu / h) * eccentricity * np.sin(nu)
        v_t = h / r
        v_x_orb = v_r * np.cos(nu) - v_t * np.sin(nu)
        v_y_orb = v_r * np.sin(nu) + v_t * np.cos(nu)
        v_z_orb = 0.0

        # Rotate to 3D space
        cos_apo = np.cos(arg_periapsis)
        sin_apo = np.sin(arg_periapsis)
        cos_lan = np.cos(long_asc_node)
        sin_lan = np.sin(long_asc_node)
        cos_inc = np.cos(inclination)
        sin_inc = np.sin(inclination)

        # Rotation matrix
        x = (cos_lan * (cos_apo * x_orb - sin_apo * y_orb) -
             sin_lan * sin_inc * (sin_apo * x_orb + cos_apo * y_orb))
        y = (sin_lan * (cos_apo * x_orb - sin_apo * y_orb) +
             cos_lan * sin_inc * (sin_apo * x_orb + cos_apo * y_orb))
        z = cos_inc * (sin_apo * x_orb + cos_apo * y_orb)

        vx = (cos_lan * (cos_apo * v_x_orb - sin_apo * v_y_orb) -
              sin_lan * sin_inc * (sin_apo * v_x_orb + cos_apo * v_y_orb))
        vy = (sin_lan * (cos_apo * v_x_orb - sin_apo * v_y_orb) +
              cos_lan * sin_inc * (sin_apo * v_x_orb + cos_apo * v_y_orb))
        vz = cos_inc * (sin_apo * v_x_orb + cos_apo * v_y_orb)

        position = np.array([x, y, z])
        velocity = np.array([vx, vy, vz])

        return position, velocity


class NBodySimulator:
    """N-body gravitational simulator"""

    def __init__(self, G: float = 1.0):
        """
        Initialize simulator

        Args:
            G: Gravitational constant (normalized)
        """
        self.G = G

    def compute_accelerations(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gravitational accelerations via pairwise forces

        Args:
            positions: (N, 3) particle positions
            masses: (N,) particle masses

        Returns:
            accelerations: (N, 3) accelerations
        """
        N = len(masses)
        accelerations = np.zeros((N, 3))

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # Vector from i to j
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec) + 1e-10  # Avoid division by zero

                # Acceleration from body j on i
                a_mag = self.G * masses[j] / (r_mag ** 3)
                accelerations[i] += a_mag * r_vec

        return accelerations

    def step_verlet(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        masses: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Velocity Verlet integration step

        Args:
            positions: (N, 3) current positions
            velocities: (N, 3) current velocities
            accelerations: (N, 3) current accelerations
            masses: (N,) particle masses
            dt: Time step

        Returns:
            new_positions: (N, 3)
            new_velocities: (N, 3)
            new_accelerations: (N, 3)
        """
        # Update positions
        new_positions = positions + velocities * dt + 0.5 * accelerations * (dt ** 2)

        # Compute new accelerations
        new_accelerations = self.compute_accelerations(new_positions, masses)

        # Update velocities
        new_velocities = velocities + 0.5 * (accelerations + new_accelerations) * dt

        return new_positions, new_velocities, new_accelerations

    def simulate(
        self,
        central_mass: float,
        orbital_elements: List[Dict],
        timesteps: int = 1000,
        dt: float = 0.01,
        return_every: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate N-body system

        Args:
            central_mass: Mass of central star (normalized)
            orbital_elements: List of dicts with keys:
                'mass', 'semi_major_axis', 'eccentricity',
                'inclination', 'arg_periapsis', 'long_asc_node', 'mean_anomaly'
            timesteps: Number of integration steps
            dt: Time step size
            return_every: Return trajectory every N steps

        Returns:
            trajectory: (timesteps//return_every, N+1, 3) positions over time
            masses: (N+1,) array of masses [central, planets...]
        """
        # Build mass array (planet masses in solar units)
        earth_to_solar = 3.003e-6
        sim_elements = []
        for elem in orbital_elements:
            sim_elem = dict(elem)
            sim_elem['mass'] = elem['mass'] * earth_to_solar
            sim_elements.append(sim_elem)

        masses = np.array([central_mass] + [elem['mass'] for elem in sim_elements])
        N = len(masses)

        # Initialize positions and velocities
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        # Central star at origin
        positions[0] = np.array([0.0, 0.0, 0.0])
        velocities[0] = np.array([0.0, 0.0, 0.0])

        # Place planets in orbits
        for i, elem in enumerate(sim_elements):
            pos, vel = OrbitalMechanics.orbital_to_cartesian(
                elem['semi_major_axis'],
                elem['eccentricity'],
                elem['inclination'],
                elem['arg_periapsis'],
                elem['long_asc_node'],
                elem['mean_anomaly'],
                mu=self.G * central_mass,
            )
            positions[i + 1] = pos
            velocities[i + 1] = vel

        # Center of mass correction to prevent drift
        total_mass = np.sum(masses)
        if total_mass > 0:
            pos_cm = np.einsum("i,ij->j", masses, positions) / total_mass
            vel_cm = np.einsum("i,ij->j", masses, velocities) / total_mass
            positions = positions - pos_cm
            velocities = velocities - vel_cm

        # Compute initial accelerations
        accelerations = self.compute_accelerations(positions, masses)

        # Storage
        trajectory = []
        trajectory.append(positions.copy())

        # Integration loop
        for step in range(1, timesteps):
            positions, velocities, accelerations = self.step_verlet(
                positions, velocities, accelerations, masses, dt
            )

            if step % return_every == 0:
                trajectory.append(positions.copy())

        return np.array(trajectory), masses
