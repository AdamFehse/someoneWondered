/**
 * Space Tycoon — Game Engine
 *
 * Manages tycoon mode state, game loop, resource ticking,
 * colonization, events, and scoring.
 */

// ── Planet Types ──────────────────────────────────────────────
export const PLANET_TYPES = {
  ROCKY: {
    id: 'rocky',
    label: 'Rocky',
    emoji: '🪨',
    habitability: [40, 90],
    primaryResource: 'metals',
    secondaryResource: 'energy',
    rarity: 0.25,
  },
  OCEAN: {
    id: 'ocean',
    label: 'Ocean',
    emoji: '🌊',
    habitability: [50, 95],
    primaryResource: 'water',
    secondaryResource: 'crystals',
    rarity: 0.15,
  },
  DESERT: {
    id: 'desert',
    label: 'Desert',
    emoji: '🏜️',
    habitability: [20, 60],
    primaryResource: 'energy',
    secondaryResource: 'metals',
    rarity: 0.20,
  },
  ICE: {
    id: 'ice',
    label: 'Ice',
    emoji: '❄️',
    habitability: [5, 30],
    primaryResource: 'water',
    secondaryResource: 'crystals',
    rarity: 0.15,
  },
  LAVA: {
    id: 'lava',
    label: 'Lava',
    emoji: '🌋',
    habitability: [0, 10],
    primaryResource: 'energy',
    secondaryResource: 'metals',
    rarity: 0.10,
  },
  GAS_GIANT: {
    id: 'gas_giant',
    label: 'Gas Giant',
    emoji: '🪐',
    habitability: [0, 0],
    primaryResource: 'gas',
    secondaryResource: 'energy',
    rarity: 0.15,
  },
};

const PLANET_TYPE_LIST = Object.values(PLANET_TYPES);

// Star name prefixes/suffixes for procedural naming
const STAR_PREFIXES = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Kappa', 'Proxima', 'Nova', 'Vega', 'Rigel', 'Cygnus', 'Lyra', 'Draco', 'Orion', 'Cetus', 'Hydra', 'Pavo', 'Lynx', 'Mira'];
const STAR_SUFFIXES = ['Prime', 'Major', 'Minor', 'IX', 'VII', 'III', 'IV', 'II', 'XI', 'XII', 'V', 'VIII'];
const PLANET_NAMES = ['Kepler', 'Trappist', 'Gliese', 'Proxima', 'Luyten', 'Wolf', 'Ross', 'Barnards', 'Cancri', 'Pegasi', 'Virginis', 'Eridani', 'Andromedae', 'Tauri', 'Aquilae', 'Ursae', 'Sirius', 'Centauri', 'Cygni', 'Lyrae'];

// Tech tree
export const TECH_TREE = [
  {
    id: 'scanners',
    label: 'Scanners',
    emoji: '📡',
    description: 'Reveal planet stats before colonizing',
    cost: { crystals: 50 },
    effect: { revealBeforeColonize: true },
  },
  {
    id: 'stabilizers',
    label: 'Stabilizers',
    emoji: '🔧',
    description: 'Reduce orbital ejection chance by 75%',
    cost: { metals: 100, energy: 50 },
    effect: { ejectionReduction: 0.75 },
  },
  {
    id: 'advanced_mining',
    label: 'Advanced Mining',
    emoji: '⛏️',
    description: '+50% resource output from all colonies',
    cost: { metals: 80, crystals: 30 },
    effect: { incomeMultiplier: 1.5 },
  },
  {
    id: 'warp_gates',
    label: 'Warp Gates',
    emoji: '🌀',
    description: 'Instant trade travel between owned planets',
    cost: { crystals: 200, energy: 100 },
    effect: { instantTrade: true },
  },
  {
    id: 'terraforming',
    label: 'Terraforming',
    emoji: '🌍',
    description: '+20% habitability on all colonies',
    cost: { water: 150, crystals: 100 },
    effect: { habitabilityBonus: 20 },
  },
];

// Random events
export const EVENTS = [
  {
    id: 'solar_flare',
    label: 'Solar Flare',
    emoji: '☀️',
    message: 'A solar flare erupts! Unprotected colonies may lose buildings.',
    probability: 0.15,
  },
  {
    id: 'asteroid_belt',
    label: 'Asteroid Belt',
    emoji: '☄️',
    message: 'Asteroid belt discovered! Bonus metals incoming.',
    probability: 0.12,
  },
  {
    id: 'orbital_instability',
    label: 'Orbital Instability',
    emoji: '⚠️',
    message: 'Orbital instability detected! A planet may be ejected.',
    probability: 0.08,
  },
  {
    id: 'trade_windfall',
    label: 'Trade Windfall',
    emoji: '💰',
    message: 'Passing merchant fleet trades generously! Bonus credits.',
    probability: 0.10,
  },
  {
    id: 'crystal_discovery',
    label: 'Crystal Discovery',
    emoji: '💎',
    message: 'Rare crystals discovered on a planet! Temporary bonus.',
    probability: 0.10,
  },
];

// Building types
export const BUILDINGS = [
  { id: 'mine', label: 'Mine', emoji: '⛏️', cost: { metals: 30, credits: 20 }, output: { metals: 3 }, perTick: true },
  { id: 'gas_harvester', label: 'Gas Harvester', emoji: '💨', cost: { metals: 25, credits: 15 }, output: { gas: 3 }, perTick: true },
  { id: 'habitat', label: 'Habitat', emoji: '🏠', cost: { metals: 40, credits: 30 }, output: { population: 10 }, perTick: false },
  { id: 'lab', label: 'Research Lab', emoji: '🔬', cost: { crystals: 15, credits: 25 }, output: { crystals: 2 }, perTick: true },
  { id: 'trade_hub', label: 'Trade Hub', emoji: '🏪', cost: { metals: 35, energy: 20, credits: 20 }, output: {}, perTick: false, enablesTrade: true },
  { id: 'solar_panel', label: 'Solar Panel', emoji: '⚡', cost: { metals: 20, credits: 10 }, output: { energy: 4 }, perTick: true },
];

// Starting resources
export const STARTING_RESOURCES = {
  credits: 200,
  metals: 50,
  gas: 10,
  water: 20,
  crystals: 15,
  energy: 30,
};

// Colonization cost
export const COLONIZATION_COST = {
  credits: 100,
  metals: 30,
  energy: 20,
};

// Resource tick interval (ms)
export const TICK_INTERVAL = 2000;

// ── Utilities ─────────────────────────────────────────────────
function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function randInt(min, max) {
  return Math.floor(rand(min, max + 1));
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function pickPlanetType() {
  const roll = Math.random();
  let cumulative = 0;
  for (const type of PLANET_TYPE_LIST) {
    cumulative += type.rarity;
    if (roll <= cumulative) return type;
  }
  return PLANET_TYPES.ROCKY;
}

// ── Tycoon Class ──────────────────────────────────────────────
export class SpaceTycoon {
  constructor() {
    this.active = false;
    this.paused = true;
    this.tick = 0;
    this.state = null;
    this.tickTimer = null;
    this.eventListeners = [];
    this.pendingEvent = null;
  }

  // ── Lifecycle ───────────────────────────────────────────────

  on(event, fn) {
    this.eventListeners.push({ event, fn });
  }

  emit(event, data) {
    for (const { event: e, fn } of this.eventListeners) {
      if (e === event) fn(data);
    }
  }

  /**
   * Initialize tycoon mode from a generated system.
   * Called after SpaceVisualization loads systemData.
   */
  initFromSystem(systemData, visualization) {
    const { central_mass, bodies, orbital_elements } = systemData;

    // Generate star name
    const starName = `${pick(STAR_PREFIXES)} ${pick(STAR_SUFFIXES)}`;

    // Build planet data from system
    const planets = [];
    for (let i = 1; i < bodies.length; i++) {
      const body = bodies[i];
      const mass = body.mass;
      const orbitalElem = orbital_elements ? orbital_elements[i - 1] : null;
      const planetType = pickPlanetType();

      // Habitability based on type + distance from star (closer = hotter)
      let habitability = randInt(planetType.habitability[0], planetType.habitability[1]);

      // Stability: based on eccentricity (lower = more stable)
      let stability = 100;
      if (orbitalElem) {
        const ecc = orbitalElem.eccentricity || 0;
        stability = Math.max(5, Math.round(100 - ecc * 80));
      }

      const planet = {
        id: i,
        name: `${pick(PLANET_NAMES)}-${randInt(100, 999)}`,
        type: planetType,
        color: visualization.bodies[i] ? visualization.bodies[i].color : 0x4a90e2,
        mass,
        orbital: orbitalElem ? {
          semi_major_axis: orbital_elem.semi_major_axis || 1,
          eccentricity: orbital_elem.eccentricity || 0,
          inclination: orbital_elem.inclination || 0,
          longitude_ascending_node: orbital_elem.longitude_ascending_node || orbital_elem.long_ascending_node || 0,
          argument_periapsis: orbital_elem.argument_periapsis || orbital_elem.arg_periapsis || 0,
          mean_anomaly: orbital_elem.mean_anomaly || 0,
        } : null,
        stats: {
          habitability,
          stability,
          primaryResource: planetType.primaryResource,
          secondaryResource: planetType.secondaryResource,
        },
        colony: null, // { owned, buildings[], population, level } when colonized
      };

      planets.push(planet);
    }

    this.state = {
      version: 1,
      star: { name: starName, mass: central_mass, type: this._starType(central_mass) },
      planets,
      resources: { ...STARTING_RESOURCES },
      tech: [],
      ships: [],
      score: { total: 0, colonies: 0, tradeRoutes: 0, survivalTime: 0 },
      tick: 0,
      paused: true,
      events: [],
      bonus: {
        incomeMultiplier: 1,
        ejectionReduction: 0,
        habitabilityBonus: 0,
        revealBeforeColonize: false,
        instantTrade: false,
      },
    };

    this.tick = 0;
    this.active = true;
    this.emit('init', this.state);
    return this.state;
  }

  _starType(mass) {
    if (mass < 0.3) return { class: 'M', label: 'Red Dwarf', color: 0xff6644 };
    if (mass < 0.8) return { class: 'K', label: 'Orange Dwarf', color: 0xffaa44 };
    if (mass < 1.2) return { class: 'G', label: 'Yellow Dwarf', color: 0xffdd44 };
    if (mass < 2.0) return { class: 'F', label: 'Yellow-White', color: 0xffffcc };
    return { class: 'A', label: 'White Star', color: 0xccddff };
  }

  start() {
    if (!this.state) return;
    this.paused = false;
    this.state.paused = false;
    this._startTickLoop();
    this.emit('start', this.state);
  }

  pause() {
    if (!this.state) return;
    this.paused = true;
    this.state.paused = true;
    this._stopTickLoop();
    this.emit('pause', this.state);
  }

  togglePause() {
    if (this.paused) this.start();
    else this.pause();
  }

  _startTickLoop() {
    this._stopTickLoop();
    this.tickTimer = setInterval(() => this._tick(), TICK_INTERVAL);
  }

  _stopTickLoop() {
    if (this.tickTimer) {
      clearInterval(this.tickTimer);
      this.tickTimer = null;
    }
  }

  // ── Game Tick ───────────────────────────────────────────────

  _tick() {
    if (!this.state || this.paused) return;

    this.tick++;
    this.state.tick = this.tick;

    // Passive income from colonies
    this._collectResources();

    // Advance trade ships
    this._advanceShips();

    // Random events (check every 10 ticks)
    if (this.tick % 10 === 0) {
      this._checkEvents();
    }

    // Stability check (every 20 ticks)
    if (this.tick % 20 === 0) {
      this._checkStability();
    }

    // Score: survival time
    this.state.score.survivalTime = this.tick;

    this.emit('tick', this.state);
  }

  _collectResources() {
    const { planets, resources, bonus } = this.state;
    for (const planet of planets) {
      if (!planet.colony) continue;

      const multiplier = bonus.incomeMultiplier;
      const habMod = 1 + (planet.stats.habitability + bonus.habitabilityBonus) / 200;

      // Primary resource
      const primary = planet.stats.primaryResource;
      const primaryAmount = Math.round(2 * multiplier * habMod * (planet.colony.buildings.length + 1));
      resources[primary] = (resources[primary] || 0) + primaryAmount;

      // Secondary resource (half rate)
      const secondary = planet.stats.secondaryResource;
      const secondaryAmount = Math.round(1 * multiplier * habMod * (planet.colony.buildings.length + 1));
      resources[secondary] = (resources[secondary] || 0) + secondaryAmount;

      // Building output
      for (const bId of planet.colony.buildings) {
        const building = BUILDINGS.find(b => b.id === bId);
        if (building && building.perTick && building.output) {
          for (const [res, amount] of Object.entries(building.output)) {
            if (res === 'population') {
              planet.colony.population += amount;
            } else {
              resources[res] = (resources[res] || 0) + amount * multiplier;
            }
          }
        }
      }
    }
  }

  _advanceShips() {
    const { ships, planets, bonus } = this.state;
    for (let i = ships.length - 1; i >= 0; i--) {
      const ship = ships[i];
      if (bonus.instantTrade) {
        // Instant delivery
        const toPlanet = planets.find(p => p.id === ship.to);
        if (toPlanet && toPlanet.colony) {
          for (const [res, amount] of Object.entries(ship.cargo)) {
            this.state.resources[res] = (this.state.resources[res] || 0) + amount;
          }
        }
        ships.splice(i, 1);
        this.emit('shipArrived', ship);
      } else {
        ship.progress += 0.1; // 10 ticks to travel
        if (ship.progress >= 1) {
          const toPlanet = planets.find(p => p.id === ship.to);
          if (toPlanet && toPlanet.colony) {
            for (const [res, amount] of Object.entries(ship.cargo)) {
              this.state.resources[res] = (this.state.resources[res] || 0) + amount;
            }
          }
          ships.splice(i, 1);
          this.emit('shipArrived', ship);
        }
      }
    }
  }

  _checkEvents() {
    for (const event of EVENTS) {
      if (Math.random() < event.probability) {
        this._triggerEvent(event);
        break; // One event per check
      }
    }
  }

  _triggerEvent(event) {
    const { planets, resources, bonus } = this.state;

    switch (event.id) {
      case 'solar_flare': {
        // Damage a random owned colony without stabilizers
        const owned = planets.filter(p => p.colony);
        if (owned.length > 0) {
          const target = pick(owned);
          if (Math.random() > bonus.ejectionReduction) {
            // Remove a random building
            if (target.colony.buildings.length > 0) {
              const removed = target.colony.buildings.pop();
              this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} Lost ${removed} on ${target.name}.`, severity: 'bad' });
            } else {
              this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} No buildings damaged on ${target.name} — colony too new.`, severity: 'warning' });
            }
          } else {
            this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} Stabilizers protected your colonies!`, severity: 'good' });
          }
        }
        break;
      }
      case 'asteroid_belt': {
        resources.metals = (resources.metals || 0) + 50 + randInt(10, 40);
        this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} +${resources.metals} metals collected.`, severity: 'good' });
        break;
      }
      case 'orbital_instability': {
        const unowned = planets.filter(p => !p.colony);
        const owned = planets.filter(p => p.colony);
        if (owned.length > 0 && Math.random() < (1 - bonus.ejectionReduction)) {
          const target = pick(owned);
          target.stats.stability = Math.max(5, target.stats.stability - randInt(15, 35));
          this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} ${target.name} stability dropped to ${target.stats.stability}%.`, severity: 'warning' });
        } else if (unowned.length > 0 && Math.random() < 0.3 * (1 - bonus.ejectionReduction)) {
          const target = pick(unowned);
          target.stats.stability = 0;
          this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} ${target.name} has been ejected from the system!`, severity: 'bad' });
        } else {
          this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} Stabilizers held the system together.`, severity: 'good' });
        }
        break;
      }
      case 'trade_windfall': {
        const bonusCredits = randInt(50, 150);
        resources.credits = (resources.credits || 0) + bonusCredits;
        this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} +${bonusCredits} credits!`, severity: 'good' });
        break;
      }
      case 'crystal_discovery': {
        const owned = planets.filter(p => p.colony);
        if (owned.length > 0) {
          const target = pick(owned);
          const bonusCrystals = randInt(15, 40);
          resources.crystals = (resources.crystals || 0) + bonusCrystals;
          this.state.events.push({ tick: this.tick, id: event.id, message: `${event.message} +${bonusCrystals} crystals on ${target.name}!`, severity: 'good' });
        }
        break;
      }
    }

    this.emit('event', { event, state: this.state });
  }

  _checkStability() {
    const { planets, bonus } = this.state;
    for (const planet of planets) {
      if (!planet.colony) continue;
      if (planet.stats.stability <= 5) {
        // Planet ejected!
        this.state.events.push({
          tick: this.tick,
          id: 'ejection',
          message: `${planet.name} has been ejected from the system! Colony lost!`,
          severity: 'bad',
        });
        planet.colony = null;
        this.state.score.colonies = Math.max(0, this.state.score.colonies - 1);
        this.emit('planetEjected', { planet, state: this.state });
      }
    }
  }

  // ── Actions ─────────────────────────────────────────────────

  canAfford(cost) {
    if (!this.state) return false;
    for (const [res, amount] of Object.entries(cost)) {
      if ((this.state.resources[res] || 0) < amount) return false;
    }
    return true;
  }

  spend(cost) {
    if (!this.canAfford(cost)) return false;
    for (const [res, amount] of Object.entries(cost)) {
      this.state.resources[res] -= amount;
    }
    return true;
  }

  colonizePlanet(planetId) {
    if (!this.state) return { success: false, error: 'No game active' };
    const planet = this.state.planets.find(p => p.id === planetId);
    if (!planet) return { success: false, error: 'Planet not found' };
    if (planet.colony) return { success: false, error: 'Already colonized' };

    if (!this.spend(COLONIZATION_COST)) {
      return { success: false, error: 'Not enough resources' };
    }

    planet.colony = {
      owned: true,
      buildings: [],
      population: 5,
      level: 1,
      foundedTick: this.tick,
    };

    this.state.score.colonies++;
    this.state.score.total += 100;

    this.emit('colonize', { planet, state: this.state });
    return { success: true, planet };
  }

  buildOnPlanet(planetId, buildingId) {
    if (!this.state) return { success: false, error: 'No game active' };
    const planet = this.state.planets.find(p => p.id === planetId);
    if (!planet || !planet.colony) return { success: false, error: 'No colony' };

    const building = BUILDINGS.find(b => b.id === buildingId);
    if (!building) return { success: false, error: 'Unknown building' };

    if (!this.spend(building.cost)) {
      return { success: false, error: 'Not enough resources' };
    }

    // Gas harvesters only on gas giants
    if (building.id === 'gas_harvester' && planet.type.id !== 'gas_giant') {
      // Refund
      for (const [res, amount] of Object.entries(building.cost)) {
        this.state.resources[res] += amount;
      }
      return { success: false, error: 'Gas harvesters only work on gas giants' };
    }

    // Trade hubs: one per planet
    if (building.id === 'trade_hub' && planet.colony.buildings.includes('trade_hub')) {
      for (const [res, amount] of Object.entries(building.cost)) {
        this.state.resources[res] += amount;
      }
      return { success: false, error: 'Trade hub already built' };
    }

    planet.colony.buildings.push(buildingId);
    this.state.score.total += 25;
    this.emit('build', { planet, building, state: this.state });
    return { success: true, planet, building };
  }

  unlockTech(techId) {
    if (!this.state) return { success: false, error: 'No game active' };
    if (this.state.tech.includes(techId)) return { success: false, error: 'Already unlocked' };

    const tech = TECH_TREE.find(t => t.id === techId);
    if (!tech) return { success: false, error: 'Unknown tech' };

    if (!this.spend(tech.cost)) {
      return { success: false, error: 'Not enough resources' };
    }

    this.state.tech.push(techId);

    // Apply effects
    const bonus = this.state.bonus;
    if (tech.effect.incomeMultiplier) bonus.incomeMultiplier *= tech.effect.incomeMultiplier;
    if (tech.effect.ejectionReduction) bonus.ejectionReduction = Math.min(0.95, bonus.ejectionReduction + tech.effect.ejectionReduction);
    if (tech.effect.habitabilityBonus) bonus.habitabilityBonus += tech.effect.habitabilityBonus;
    if (tech.effect.revealBeforeColonize) bonus.revealBeforeColonize = true;
    if (tech.effect.instantTrade) bonus.instantTrade = true;

    this.state.score.total += 50;
    this.emit('tech', { tech, state: this.state });
    return { success: true, tech };
  }

  sendTradeShip(fromId, toId) {
    if (!this.state) return { success: false, error: 'No game active' };
    const fromPlanet = this.state.planets.find(p => p.id === fromId);
    const toPlanet = this.state.planets.find(p => p.id === toId);
    if (!fromPlanet?.colony || !toPlanet?.colony) return { success: false, error: 'Both planets need colonies' };
    if (!fromPlanet.colony.buildings.includes('trade_hub')) return { success: false, error: 'Need a trade hub on source planet' };

    // Cargo: send surplus primary resource
    const cargo = {};
    const res = fromPlanet.stats.primaryResource;
    const amount = Math.min(10, Math.floor((this.state.resources[res] || 0) * 0.2));
    if (amount <= 0) return { success: false, error: 'No surplus to trade' };

    cargo[res] = amount;
    this.state.resources[res] -= amount;

    const ship = {
      from: fromId,
      to: toId,
      progress: 0,
      cargo,
    };

    this.state.ships.push(ship);

    if (this.state.score.tradeRoutes === 0 || !this._hasTradeRoute(fromId, toId)) {
      this.state.score.tradeRoutes++;
    }

    this.emit('shipSent', { ship, fromPlanet, toPlanet, state: this.state });
    return { success: true, ship };
  }

  _hasTradeRoute(fromId, toId) {
    return this.state.ships.some(s =>
      (s.from === fromId && s.to === toId) || (s.from === toId && s.to === fromId)
    );
  }

  // ── Save / Load ─────────────────────────────────────────────

  save() {
    if (!this.state) return false;
    try {
      localStorage.setItem('someoneWondered-tycoon', JSON.stringify(this.state));
      return true;
    } catch (e) {
      console.warn('Failed to save tycoon game:', e);
      return false;
    }
  }

  load() {
    try {
      const raw = localStorage.getItem('someoneWondered-tycoon');
      if (!raw) return false;
      this.state = JSON.parse(raw);
      this.tick = this.state.tick || 0;
      this.paused = this.state.paused !== false;
      this.active = true;
      if (!this.paused) this._startTickLoop();
      this.emit('load', this.state);
      return true;
    } catch (e) {
      console.warn('Failed to load tycoon game:', e);
      return false;
    }
  }

  hasSave() {
    return !!localStorage.getItem('someoneWondered-tycoon');
  }

  deleteSave() {
    localStorage.removeItem('someoneWondered-tycoon');
    this.state = null;
    this.tick = 0;
    this.active = false;
    this._stopTickLoop();
  }

  // ── Scoring ─────────────────────────────────────────────────

  getScore() {
    if (!this.state) return 0;
    const s = this.state.score;
    let total = s.total;
    total += s.colonies * 100;
    total += s.tradeRoutes * 10;
    total += s.survivalTime;
    // Bonuses
    if (this.state.planets.every(p => p.colony)) total += 500;
    if (this.tick >= 100) total += 200;
    return total;
  }

  // ── Cleanup ─────────────────────────────────────────────────

  destroy() {
    this._stopTickLoop();
    this.eventListeners = [];
    this.state = null;
    this.active = false;
    this.tick = 0;
  }
}
