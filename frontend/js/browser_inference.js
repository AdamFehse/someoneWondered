/**
 * ONNX Runtime Web inference for browser-side generation.
 */
import { API_DEFAULTS, INFERENCE, MODEL } from "./constants.js";
import { generateOrbitTrajectory } from "./orbit_simulation.js";

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / sum);
}

function sampleFromProbs(probs) {
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < probs.length; i += 1) {
    acc += probs[i];
    if (r <= acc) {
      return i;
    }
  }
  return probs.length - 1;
}

function applyTopK(logits, topK) {
  if (!topK || topK <= 0 || topK >= logits.length) {
    return logits.slice();
  }
  const indices = logits
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, topK)
    .map((item) => item.index);
  const filtered = new Array(logits.length).fill(-Infinity);
  for (const idx of indices) {
    filtered[idx] = logits[idx];
  }
  return filtered;
}

function tokensToBigInt64(tokens) {
  const arr = new BigInt64Array(tokens.length);
  for (let i = 0; i < tokens.length; i += 1) {
    arr[i] = BigInt(tokens[i]);
  }
  return arr;
}

export class BrowserInference {
  constructor(options = {}) {
    this.modelUrl = options.modelUrl || MODEL.ONNX_MODEL_URL;
    this.statsUrl = options.statsUrl || MODEL.STATS_URL;
    this.wasmBaseUrl = options.wasmBaseUrl || MODEL.WASM_BASE_URL;
    this.vocabSize = options.vocabSize || INFERENCE.VOCAB_SIZE;
    this.maxSeqLen = options.maxSeqLen || INFERENCE.MAX_SEQ_LEN;
    this.numBins = options.numBins || INFERENCE.NUM_BINS;
    this.offset = options.offset || INFERENCE.OFFSET;
    this.session = null;
    this.stats = null;
    this.ready = false;
  }

  async init() {
    if (this.ready) {
      return;
    }
    if (typeof BigInt64Array === "undefined") {
      throw new Error("BigInt64Array not supported in this browser");
    }
    if (!window.ort) {
      throw new Error("onnxruntime-web not loaded");
    }
    const ort = window.ort;
    ort.env.wasm.wasmPaths = this.wasmBaseUrl;

    const statsResponse = await fetch(this.statsUrl);
    if (!statsResponse.ok) {
      throw new Error(`Failed to load stats: ${statsResponse.status}`);
    }
    this.stats = await statsResponse.json();

    this.session = await ort.InferenceSession.create(this.modelUrl, {
      executionProviders: ["wasm"]
    });
    this.ready = true;
  }

  dequantize(token) {
    if (token < 3) {
      return 0;
    }
    return (token - this.offset) / (this.numBins - 1);
  }

  denormMass(normValue) {
    const logMass =
      normValue * (this.stats.mass_max - this.stats.mass_min) + this.stats.mass_min;
    return Math.pow(10, logMass);
  }

  denormSemiMajorAxis(normValue) {
    const logSma =
      normValue * (this.stats.semi_axis_max - this.stats.semi_axis_min) +
      this.stats.semi_axis_min;
    return Math.pow(10, logSma);
  }

  denormEccentricity(normValue) {
    const ecc =
      normValue * (this.stats.ecc_max - this.stats.ecc_min) + this.stats.ecc_min;
    return Math.min(Math.max(ecc, 0), 0.99);
  }

  denormAngle(normValue) {
    return normValue * Math.PI * 2;
  }

  parsePlanetTokens(tokens) {
    const values = tokens.map((t) => this.dequantize(t));
    return {
      mass: this.denormMass(values[0]),
      semi_major_axis: this.denormSemiMajorAxis(values[1]),
      eccentricity: this.denormEccentricity(values[2]),
      inclination: this.denormAngle(values[3]),
      arg_periapsis: this.denormAngle(values[4]),
      long_asc_node: 0,
      mean_anomaly: 0
    };
  }

  stabilizePlanets(planets) {
    for (const planet of planets) {
      planet.semi_major_axis = Math.max(planet.semi_major_axis, 0.05);
      planet.eccentricity = Math.min(planet.eccentricity, 0.3);
      planet.mean_anomaly = Math.random() * Math.PI * 2;
      planet.long_asc_node = Math.random() * Math.PI * 2;
    }

    if (planets.length > 1) {
      planets.sort((a, b) => a.semi_major_axis - b.semi_major_axis);
      const minRatio = 1.3;
      for (let i = 1; i < planets.length; i += 1) {
        const prev = planets[i - 1].semi_major_axis;
        if (planets[i].semi_major_axis < prev * minRatio) {
          planets[i].semi_major_axis = prev * minRatio;
        }
      }
      for (let i = 0; i < planets.length; i += 1) {
        const phase = (2 * Math.PI * i) / planets.length;
        planets[i].mean_anomaly = phase;
        planets[i].long_asc_node = phase;
      }
    }
  }

  async generateTokens(promptTokens, maxNewTokens, temperature, topK, onToken) {
    const ort = window.ort;
    const generated = promptTokens.slice();
    for (let i = 0; i < maxNewTokens; i += 1) {
      const inputSeq = generated.slice(-this.maxSeqLen);
      const inputTensor = new ort.Tensor(
        "int64",
        tokensToBigInt64(inputSeq),
        [1, inputSeq.length]
      );
      const outputs = await this.session.run({ input_ids: inputTensor });
      const output = outputs.logits || outputs.output || outputs[Object.keys(outputs)[0]];
      const logits = output.data;
      const vocabSize = this.vocabSize;
      const offset = (inputSeq.length - 1) * vocabSize;
      const lastLogits = Array.from(logits.slice(offset, offset + vocabSize));
      const scaled = lastLogits.map((v) => v / Math.max(temperature, 0.01));
      const filtered = applyTopK(scaled, topK);
      const probs = softmax(filtered);
      const nextToken = sampleFromProbs(probs);
      generated.push(nextToken);
      if (onToken) {
        await onToken(generated, i + 1);
      }
    }
    return generated;
  }

  async generateSystem(params = {}, options = {}) {
    await this.init();

    const onProgress = options.onProgress;
    const request = {
      central_mass: params.central_mass ?? API_DEFAULTS.CENTRAL_MASS,
      num_bodies: params.num_bodies ?? API_DEFAULTS.NUM_BODIES,
      temperature: params.temperature ?? API_DEFAULTS.TEMPERATURE,
      top_k: params.top_k ?? API_DEFAULTS.TOP_K,
      simulation_timesteps:
        params.simulation_timesteps ?? API_DEFAULTS.SIMULATION_TIMESTEPS,
      simulation_dt: params.simulation_dt ?? API_DEFAULTS.SIMULATION_DT
    };

    const centralMassToken = Math.round(
      request.central_mass * (this.numBins - 1) + this.offset
    );
    const numBodiesToken = request.num_bodies + this.offset;
    const prompt = [0, centralMassToken, numBodiesToken];
    const maxTokens = request.num_bodies * 5 + 2;

    const denormCentralMass = this.denormMass(this.dequantize(prompt[1]));
    let lastPlanetCount = 0;
    const generated = await this.generateTokens(
      prompt,
      maxTokens,
      request.temperature,
      request.top_k,
      async (tokens) => {
        if (!onProgress) {
          return;
        }
        const availablePlanets = Math.floor((tokens.length - 3) / 5);
        if (availablePlanets <= lastPlanetCount) {
          return;
        }
        const planets = [];
        let idx = 3;
        for (let i = 0; i < availablePlanets; i += 1) {
          const planetTokens = tokens.slice(idx, idx + 5);
          planets.push(this.parsePlanetTokens(planetTokens));
          idx += 5;
        }
        this.stabilizePlanets(planets);
        const { trajectory, bodies } = generateOrbitTrajectory(
          denormCentralMass,
          planets,
          request.simulation_timesteps,
          request.simulation_dt
        );
        await onProgress(
          {
            central_mass: denormCentralMass,
            bodies,
            trajectory,
            orbital_elements: planets
          },
          {
            completedPlanets: availablePlanets,
            totalPlanets: request.num_bodies
          }
        );
        lastPlanetCount = availablePlanets;
      }
    );

    let idx = 1;
    const finalCentralMass = this.denormMass(this.dequantize(generated[idx]));
    idx += 2;

    const planets = [];
    for (let i = 0; i < request.num_bodies; i += 1) {
      if (idx + 5 > generated.length) {
        break;
      }
      const planetTokens = generated.slice(idx, idx + 5);
      planets.push(this.parsePlanetTokens(planetTokens));
      idx += 5;
    }

    this.stabilizePlanets(planets);

    const { trajectory, bodies } = generateOrbitTrajectory(
      finalCentralMass,
      planets,
      request.simulation_timesteps,
      request.simulation_dt
    );

    return {
      central_mass: finalCentralMass,
      bodies,
      trajectory,
      orbital_elements: planets
    };
  }
}
