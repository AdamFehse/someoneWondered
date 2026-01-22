/**
 * Lightweight orbit simulation for browser inference mode.
 * Uses two-body Keplerian orbits around a fixed central mass.
 */

const TWO_PI = Math.PI * 2;

function solveKepler(meanAnomaly, eccentricity) {
  // Newton-Raphson solve for eccentric anomaly.
  let E = eccentricity < 0.8 ? meanAnomaly : Math.PI;
  for (let i = 0; i < 6; i += 1) {
    const f = E - eccentricity * Math.sin(E) - meanAnomaly;
    const fPrime = 1 - eccentricity * Math.cos(E);
    E -= f / fPrime;
  }
  return E;
}

function rotatePosition(x, y, z, omega, inc, Omega) {
  // Rotation: Rz(Omega) * Rx(inc) * Rz(omega)
  const cosO = Math.cos(Omega);
  const sinO = Math.sin(Omega);
  const cosI = Math.cos(inc);
  const sinI = Math.sin(inc);
  const cosw = Math.cos(omega);
  const sinw = Math.sin(omega);

  const x1 = x * cosw - y * sinw;
  const y1 = x * sinw + y * cosw;

  const x2 = x1;
  const y2 = y1 * cosI - z * sinI;
  const z2 = y1 * sinI + z * cosI;

  return {
    x: x2 * cosO - y2 * sinO,
    y: x2 * sinO + y2 * cosO,
    z: z2
  };
}

export function generateOrbitTrajectory(centralMass, orbitalElements, timesteps, dt) {
  const trajectories = new Array(timesteps);
  const bodies = new Array(orbitalElements.length + 1);

  for (let t = 0; t < timesteps; t += 1) {
    const time = t * dt;
    const frame = new Array(orbitalElements.length + 1);
    frame[0] = [0, 0, 0];

    for (let i = 0; i < orbitalElements.length; i += 1) {
      const elem = orbitalElements[i];
      const a = Math.max(elem.semi_major_axis, 0.0001);
      const e = Math.min(Math.max(elem.eccentricity, 0), 0.99);
      const inc = elem.inclination || 0;
      const omega = elem.arg_periapsis || 0;
      const Omega = elem.long_asc_node || 0;
      const meanAnomaly0 = elem.mean_anomaly || 0;

      const mu = Math.max(centralMass, 0.0001);
      const n = Math.sqrt(mu / (a * a * a));
      const M = (meanAnomaly0 + n * time) % TWO_PI;
      const E = solveKepler(M, e);
      const r = a * (1 - e * Math.cos(E));
      const trueAnomaly = 2 * Math.atan2(
        Math.sqrt(1 + e) * Math.sin(E / 2),
        Math.sqrt(1 - e) * Math.cos(E / 2)
      );

      const xOrb = r * Math.cos(trueAnomaly);
      const yOrb = r * Math.sin(trueAnomaly);
      const rotated = rotatePosition(xOrb, yOrb, 0, omega, inc, Omega);

      frame[i + 1] = [rotated.x, rotated.y, rotated.z];
    }

    trajectories[t] = frame;
  }

  bodies[0] = { mass: centralMass, position: trajectories[0][0], velocity: [0, 0, 0] };
  for (let i = 0; i < orbitalElements.length; i += 1) {
    bodies[i + 1] = {
      mass: orbitalElements[i].mass,
      position: trajectories[0][i + 1],
      velocity: [0, 0, 0]
    };
  }

  return { trajectory: trajectories, bodies };
}
