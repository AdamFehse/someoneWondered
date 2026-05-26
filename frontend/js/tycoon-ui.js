/**
 * Space Tycoon — UI Renderer
 *
 * Renders tycoon UI panels: resource bar, planet info, tech tree,
 * event log, and score. Listens to tycoon events and updates DOM.
 */

import { BUILDINGS, TECH_TREE, COLONIZATION_COST } from './tycoon.js';

const RESOURCE_ICONS = {
  credits: '💰',
  metals: '⛏️',
  gas: '💨',
  water: '💧',
  crystals: '💎',
  energy: '⚡',
};

const RESOURCE_LABELS = {
  credits: 'Credits',
  metals: 'Metals',
  gas: 'Gas',
  water: 'Water',
  crystals: 'Crystals',
  energy: 'Energy',
};

export class TycoonUI {
  constructor(tycoon) {
    this.tycoon = tycoon;
    this.container = null;
    this.selectedPlanetId = null;
    this._bound = {};
  }

  /**
   * Create the tycoon UI overlay in the given container.
   * Returns the root element.
   */
  create(containerEl) {
    this.container = document.createElement('div');
    this.container.id = 'tycoon-ui';
    this.container.className = 'tycoon-ui hidden';
    this.container.innerHTML = this._template();
    containerEl.appendChild(this.container);

    // Cache DOM refs
    this.refs = {
      resourceBar: this.container.querySelector('#tycoon-resources'),
      scoreDisplay: document.getElementById('score-display'),
      cmdPlanet: this.container.querySelector('#cmd-planet'),
      cmdActions: this.container.querySelector('#cmd-actions'),
      cmdTech: this.container.querySelector('#cmd-tech'),
      cmdLogs: this.container.querySelector('#cmd-logs'),
    };

    // Wire up static event listeners
    this._bindEvents();

    // Listen to tycoon events
    this.tycoon.on('tick', () => this._renderResources());
    this.tycoon.on('colonize', (d) => this._onColonize(d));
    this.tycoon.on('build', (d) => this._onBuild(d));
    this.tycoon.on('tech', (d) => this._onTech(d));
    this.tycoon.on('event', (d) => this._onEvent(d));
    this.tycoon.on('shipSent', (d) => this._onShipSent(d));
    this.tycoon.on('shipArrived', (d) => this._onShipArrived(d));
    this.tycoon.on('planetEjected', (d) => this._onPlanetEjected(d));

    return this.container;
  }

  _template() {
    return `
      <div id="tycoon-resources" class="tycoon-resources">
        <div class="res-item" data-res="credits"><span class="res-icon">💰</span><span class="res-val">0</span></div>
        <div class="res-item" data-res="metals"><span class="res-icon">⛏️</span><span class="res-val">0</span></div>
        <div class="res-item" data-res="gas"><span class="res-icon">💨</span><span class="res-val">0</span></div>
        <div class="res-item" data-res="water"><span class="res-icon">💧</span><span class="res-val">0</span></div>
        <div class="res-item" data-res="crystals"><span class="res-icon">💎</span><span class="res-val">0</span></div>
        <div class="res-item" data-res="energy"><span class="res-icon">⚡</span><span class="res-val">0</span></div>
      </div>

      <div id="tycoon-planet-list" class="tycoon-planet-list">
        <div class="panel-header"><span>🪐 System</span></div>
        <div class="planet-list-body"></div>
      </div>

      <div id="tycoon-command-bar" class="tycoon-command-bar">
        <div id="cmd-planet" class="cmd-section"></div>
        <div id="cmd-actions" class="cmd-section"></div>
        <div id="cmd-tech" class="cmd-section"></div>
        <div id="cmd-logs" class="cmd-section"></div>
      </div>`;
  }

  _bindEvents() {
    // Nothing to bind at container level — panels wire their own buttons
  }

  show() {
    this.container?.classList.remove('hidden');
    if (this.refs.scoreDisplay) this.refs.scoreDisplay.classList.remove('hidden');
    this._renderAll();
  }

  hide() {
    this.container?.classList.add('hidden');
    if (this.refs.scoreDisplay) this.refs.scoreDisplay.classList.add('hidden');
  }

  // ── Rendering ──────────────────────────────────────────────

  _renderAll() {
    this._renderResources();
    this._renderPlanetList();
    this._renderScore();
    this._renderCommandBar();
  }

  _renderCommandBar() {
    this._renderPlanetInfo();
    this._renderActions();
    this._renderTech();
    this._renderLogs();
  }

  _renderResources() {
    if (!this.refs.resourceBar || !this.tycoon.state) return;
    const res = this.tycoon.state.resources;
    const items = this.refs.resourceBar.querySelectorAll('.res-item');
    for (const item of items) {
      const key = item.dataset.res;
      const valEl = item.querySelector('.res-val');
      if (valEl) valEl.textContent = Math.floor(res[key] || 0);
    }
  }

  _renderPlanetList() {
    if (!this.container || !this.tycoon.state) return;
    const body = this.container.querySelector('.planet-list-body');
    if (!body) return;

    const planets = this.tycoon.state.planets;
    const star = this.tycoon.state.star;

    let html = `<div class="planet-list-star"><span class="star-class">${star.type.class}</span> ${star.name}</div>`;

    for (const planet of planets) {
      const selected = this.selectedPlanetId === planet.id ? ' selected' : '';
      const colonized = planet.colony ? ' colonized' : '';
      const ejected = planet.stats.stability <= 0 ? ' ejected' : '';
      html += `
        <div class="planet-list-item${selected}${colonized}${ejected}" data-planet="${planet.id}">
          <span class="planet-dot" style="color:#${planet.color.toString(16).padStart(6, '0')}">${planet.type.emoji}</span>
          <span class="planet-list-name">${planet.name}</span>
          <span class="planet-list-type">${planet.type.label}</span>
          ${planet.colony ? '<span class="planet-colony-badge">🏛</span>' : ''}
          ${planet.stats.stability <= 0 ? '<span class="planet-ejected-badge">💥</span>' : ''}
        </div>`;
    }

    body.innerHTML = html;

    // Wire click + hover handlers
    body.querySelectorAll('.planet-list-item').forEach(el => {
      const id = parseInt(el.dataset.planet);
      const isPlanetIdx = !isNaN(id) && id > 0;

      el.addEventListener('click', () => {
        if (isPlanetIdx) {
          this.selectPlanet(id);
          if (window._visualization && window._visualization.selection.bodyIndex !== id) {
            window._visualization.selectBody(id);
          }
        }
      });
    });
  }

  _renderScore() {
    if (!this.refs.scoreDisplay) return;
    const score = this.tycoon.getScore();
    const s = this.tycoon.state?.score;
    this.refs.scoreDisplay.innerHTML = `
      <span class="score-main">⭐ ${score}</span>
      <span class="score-detail">🏛${s?.colonies || 0} 🔄${s?.tradeRoutes || 0} ⏱T${this.tycoon.tick}</span>
    `;
  }

  selectPlanet(planetId) {
    this.selectedPlanetId = planetId;
    this._renderCommandBar();
  }

  deselectPlanet() {
    this.selectedPlanetId = null;
    this._renderCommandBar();
  }

  _renderPlanetInfo() {
    const el = this.refs.cmdPlanet;
    if (!el) return;
    if (this.selectedPlanetId === null) {
      el.innerHTML = '<span class="cmd-section-label">Planet</span><span style="color:#3a4a60;font-size:11px;">Click a planet →</span>';
      return;
    }
    const planet = this.tycoon.state.planets.find(p => p.id === this.selectedPlanetId);
    if (!planet) { el.innerHTML = ''; return; }

    const habColor = planet.stats.habitability > 60 ? 'good' : planet.stats.habitability > 30 ? 'warn' : 'danger';
    const stabColor = planet.stats.stability > 60 ? 'good' : planet.stats.stability > 30 ? 'warn' : 'danger';

    el.innerHTML = `
      <span class="cmd-section-label">Planet</span>
      <span style="font-weight:700;color:#fff;font-size:12px;">${planet.type.emoji} ${planet.name}</span>
      <span style="color:#4a5a60;font-size:10px;">${planet.type.label}</span>
      <span style="margin-left:8px;font-size:10px;color:#6c7a94;">Hab</span>
      <div class="stat-bar-wrap"><div class="stat-bar-fill ${habColor}" style="width:${planet.stats.habitability}%"></div></div>
      <span style="font-size:10px;color:#6c7a94;">Stab</span>
      <div class="stat-bar-wrap"><div class="stat-bar-fill ${stabColor}" style="width:${planet.stats.stability}%"></div></div>
    `;
  }

  _renderActions() {
    const el = this.refs.cmdActions;
    if (!el) return;
    if (this.selectedPlanetId === null) {
      el.innerHTML = '<span class="cmd-section-label">Actions</span>';
      return;
    }
    const planet = this.tycoon.state.planets.find(p => p.id === this.selectedPlanetId);
    if (!planet) { el.innerHTML = ''; return; }

    let html = '<span class="cmd-section-label">Actions</span>';

    if (!planet.colony) {
      const canAfford = this.tycoon.canAfford(COLONIZATION_COST);
      html += `<button class="cmd-btn primary" data-action="colonize" ${!canAfford ? 'disabled' : ''}>🚀 Colonize</button>`;
    } else {
      // Build buttons — show 2-3 at a time
      const colony = planet.colony;
      const available = BUILDINGS.filter(b => {
        if (b.id === 'gas_harvester' && planet.type.id !== 'gas_giant') return false;
        if (b.id === 'trade_hub' && colony.buildings.includes('trade_hub')) return false;
        return !colony.buildings.includes(b.id);
      });
      for (const b of available.slice(0, 3)) {
        const canAfford = this.tycoon.canAfford(b.cost);
        const costStr = Object.entries(b.cost).map(([r, a]) => `${RESOURCE_ICONS[r]}${a}`).join(' ');
        html += `<button class="cmd-btn" data-action="build" data-building="${b.id}" ${!canAfford ? 'disabled' : ''}>${b.emoji} ${b.label}<span class="cost">${costStr}</span></button>`;
      }
      // Trade
      if (colony.buildings.includes('trade_hub')) {
        const targets = this.tycoon.state.planets.filter(p => p.id !== planet.id && p.colony);
        for (const t of targets.slice(0, 2)) {
          html += `<button class="cmd-btn" data-action="trade" data-target="${t.id}" style="border-color:rgba(255,170,0,0.2);color:#ffcc44;">🚀 → ${t.name}</button>`;
        }
      }
    }

    el.innerHTML = html;

    // Wire buttons
    el.querySelectorAll('[data-action]').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        if (action === 'colonize') {
          const r = this.tycoon.colonizePlanet(planet.id);
          if (!r.success) this._toast(r.error);
        } else if (action === 'build') {
          const r = this.tycoon.buildOnPlanet(planet.id, btn.dataset.building);
          if (!r.success) this._toast(r.error);
        } else if (action === 'trade') {
          const r = this.tycoon.sendTradeShip(planet.id, parseInt(btn.dataset.target));
          if (!r.success) this._toast(r.error);
        }
      });
    });
  }

  _renderTech() {
    const el = this.refs.cmdTech;
    if (!el) return;
    const state = this.tycoon.state;
    if (!state) return;

    let html = '<span class="cmd-section-label">Research</span>';

    for (const tech of TECH_TREE) {
      const unlocked = state.tech.includes(tech.id);
      if (unlocked) continue;
      const canAfford = this.tycoon.canAfford(tech.cost);
      const costStr = Object.entries(tech.cost).map(([r, a]) => `${RESOURCE_ICONS[r]}${a}`).join(' ');
      html += `<button class="cmd-btn${canAfford ? ' primary' : ''}" data-action="tech" data-tech="${tech.id}" ${!canAfford ? 'disabled' : ''} title="${tech.description}">${tech.emoji} ${tech.label}<span class="cost">${costStr}</span></button>`;
    }

    if (!html.includes('data-action="tech"')) {
      html += '<span style="color:#3a4a60;font-size:10px;">All researched</span>';
    }

    el.innerHTML = html;

    el.querySelectorAll('[data-action="tech"]').forEach(btn => {
      btn.addEventListener('click', () => {
        const r = this.tycoon.unlockTech(btn.dataset.tech);
        if (!r.success) this._toast(r.error);
      });
    });
  }

  _renderLogs() {
    const el = this.refs.cmdLogs;
    if (!el) return;
    const events = this.tycoon.state?.events || [];

    let html = '<span class="cmd-section-label">Log</span>';
    html += '<div style="overflow-y:auto;max-height:40px;flex:1;">';

    if (events.length === 0) {
      html += '<div style="color:#3a4a60;font-size:10px;padding:2px 0;">No events yet</div>';
    } else {
      const recent = events.slice(-5).reverse();
      for (const e of recent) {
        const cls = e.severity === 'good' ? 'log-good' : e.severity === 'bad' ? 'log-bad' : e.severity === 'warning' ? 'log-warn' : '';
        html += `<div class="log-item ${cls}"><span class="log-tick">T${e.tick}</span><span class="log-msg">${e.message}</span></div>`;
      }
    }

    html += '</div>';
    el.innerHTML = html;
  }

  // ── Event Handlers ─────────────────────────────────────────

  _onColonize(d) {
    this._toast(`🚀 Colonized ${d.planet.name}!`);
    this._renderResources();
    this._renderScore();
    this._renderCommandBar();
  }

  _onBuild(d) {
    this._toast(`🏗️ Built ${d.building.label} on ${d.planet.name}!`);
    this._renderResources();
    this._renderCommandBar();
  }

  _onTech(d) {
    this._toast(`🔬 Unlocked ${d.tech.label}!`);
    this._renderResources();
    this._renderScore();
    this._renderCommandBar();
  }

  _onEvent(d) {
    this._renderResources();
    this._renderScore();
    this._renderCommandBar();
  }

  _onShipSent(d) {
    const res = Object.entries(d.ship.cargo).map(([r, a]) => `${RESOURCE_ICONS[r]}${a}`).join(' ');
    this._toast(`🚀 Trade ship launched! Carrying ${res}`);
    this._renderResources();
  }

  _onShipArrived(d) {
    const res = Object.entries(d.ship.cargo).map(([r, a]) => `${RESOURCE_ICONS[r]}${a}`).join(' ');
    this._toast(`📦 Trade ship arrived! Delivered ${res}`);
    this._renderResources();
  }

  _onPlanetEjected(d) {
    this._toast(`💥 ${d.planet.name} was ejected!`, 4000);
    if (this.selectedPlanetId === d.planet.id) {
      this.deselectPlanet();
    }
    this._renderScore();
    this._renderPlanetList();
  }

  // ── Toast ──────────────────────────────────────────────────

  _toast(msg, duration = 2500) {
    const existing = this.container?.parentElement?.querySelector('#tycoon-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'tycoon-toast';
    toast.className = 'tycoon-toast';
    toast.textContent = msg;

    if (this.container?.parentElement) {
      this.container.parentElement.appendChild(toast);
      requestAnimationFrame(() => toast.classList.add('visible'));
      setTimeout(() => {
        toast.classList.remove('visible');
        setTimeout(() => toast.remove(), 300);
      }, duration);
    }
  }

  // ── Cleanup ────────────────────────────────────────────────

  destroy() {
    this.container?.remove();
    this.container = null;
    this.refs = {};
    this.selectedPlanetId = null;
  }
}
