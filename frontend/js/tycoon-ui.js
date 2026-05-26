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
      planetPanel: this.container.querySelector('#tycoon-planet-panel'),
      techPanel: this.container.querySelector('#tycoon-tech-panel'),
      eventLog: this.container.querySelector('#tycoon-events'),
      scoreDisplay: document.getElementById('score-display'),
      bottomBar: document.getElementById('bottom-bar'),
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

    // Make panels draggable and resizable
    this._initDraggablePanels();

    return this.container;
  }

  _initDraggablePanels() {
    const panels = this.container.querySelectorAll('.tycoon-panel:not(.tycoon-panel-static)');

    panels.forEach(panel => {
      // Make absolutely positioned
      const rect = panel.getBoundingClientRect();
      panel.style.position = 'absolute';
      panel.style.left = rect.left + 'px';
      panel.style.top = rect.top + 'px';

      const header = panel.querySelector('.panel-header');
      const body = panel.querySelector('.panel-body');
      if (!header || !body) return;

      // ── Drag ──────────────────────────────────────────────
      header.style.cursor = 'grab';

      header.addEventListener('pointerdown', (e) => {
        if (e.target.closest('button')) return; // Don't drag from buttons
        if (e.target.closest('.panel-resize')) return; // Don't drag from resize handle

        e.preventDefault();
        header.style.cursor = 'grabbing';
        panel.style.zIndex = '200';

        const startX = e.clientX;
        const startY = e.clientY;
        const origLeft = parseFloat(panel.style.left);
        const origTop = parseFloat(panel.style.top);

        const onMove = (ev) => {
          panel.style.left = (origLeft + ev.clientX - startX) + 'px';
          panel.style.top = (origTop + ev.clientY - startY) + 'px';
        };

        const onUp = () => {
          header.style.cursor = 'grab';
          panel.style.zIndex = '';
          document.removeEventListener('pointermove', onMove);
          document.removeEventListener('pointerup', onUp);
        };

        document.addEventListener('pointermove', onMove);
        document.addEventListener('pointerup', onUp);
      });

      // ── Resize ────────────────────────────────────────────
      const handle = document.createElement('div');
      handle.className = 'panel-resize';
      panel.appendChild(handle);

      handle.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        e.stopPropagation();

        const startX = e.clientX;
        const startY = e.clientY;
        const origW = panel.offsetWidth;
        const origH = panel.offsetHeight;
        const minW = 150;
        const minH = 80;

        const onMove = (ev) => {
          const w = Math.max(minW, origW + (ev.clientX - startX));
          const h = Math.max(minH, origH + (ev.clientY - startY));
          panel.style.width = w + 'px';
          panel.style.height = h + 'px';
        };

        const onUp = () => {
          document.removeEventListener('pointermove', onMove);
          document.removeEventListener('pointerup', onUp);
        };

        document.addEventListener('pointermove', onMove);
        document.addEventListener('pointerup', onUp);
      });
    });
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

      <div id="tycoon-planet-list" class="tycoon-panel tycoon-planet-list">
        <div class="panel-header"><span>🪐 System</span></div>
        <div class="panel-body planet-list-body"></div>
      </div>

      <div id="tycoon-panels" class="tycoon-panels">
        <div id="tycoon-planet-panel" class="tycoon-panel tycoon-planet-panel hidden">
          <div class="panel-header">
            <span class="panel-planet-name">—</span>
          </div>
          <div class="panel-body">
            <div class="planet-meta"></div>
            <div class="planet-colony"></div>
            <div class="planet-buildings"></div>
            <div class="planet-actions"></div>
          </div>
        </div>

        <div id="tycoon-tech-panel" class="tycoon-panel tycoon-tech-panel">
          <div class="panel-header">
            <span>🔬 Tech Tree</span>
          </div>
          <div class="panel-body tech-tree"></div>
        </div>

        <div id="tycoon-events" class="tycoon-panel tycoon-events-panel">
          <div class="panel-header">
            <span>📜 Events</span>
          </div>
          <div class="panel-body events-list"></div>
        </div>
      </div>`;
  }

  _bindEvents() {
    // Planet panel close
    const panel = this.container.querySelector('#tycoon-planet-panel');
    panel?.addEventListener('click', (e) => {
      if (e.target.classList.contains('close-panel')) {
        this.deselectPlanet();
      }
    });
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
    this._renderTech();
    this._renderEvents();
    this._renderScore();
    if (this.selectedPlanetId !== null) {
      this._renderPlanetPanel();
    }
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

  _renderTech() {
    if (!this.refs.techPanel) return;
    const container = this.refs.techPanel.querySelector('.tech-tree');
    if (!container) return;

    const state = this.tycoon.state;
    if (!state) return;

    container.innerHTML = '';

    for (const tech of TECH_TREE) {
      const unlocked = state.tech.includes(tech.id);
      const canAfford = this.tycoon.canAfford(tech.cost);

      const el = document.createElement('div');
      el.className = `tech-item${unlocked ? ' unlocked' : ''}${canAfford && !unlocked ? ' affordable' : ''}`;

      const costHtml = Object.entries(tech.cost)
        .map(([res, amt]) => `${RESOURCE_ICONS[res] || ''}${amt}`)
        .join(' ');

      el.innerHTML = `
        <div class="tech-info">
          <span class="tech-emoji">${tech.emoji}</span>
          <div>
            <div class="tech-label">${tech.label}</div>
            <div class="tech-desc">${tech.description}</div>
          </div>
        </div>
        <div class="tech-cost">${costHtml}</div>
        ${unlocked ? '<div class="tech-status">✓</div>' : `<button class="tech-unlock-btn" data-tech="${tech.id}" ${!canAfford ? 'disabled' : ''}>Unlock</button>`}
      `;

      if (!unlocked && canAfford) {
        const btn = el.querySelector('.tech-unlock-btn');
        btn.addEventListener('click', () => {
          const result = this.tycoon.unlockTech(tech.id);
          if (!result.success) {
            this._toast(result.error);
          }
        });
      }

      container.appendChild(el);
    }
  }

  _renderEvents() {
    if (!this.refs.eventLog) return;
    const list = this.refs.eventLog.querySelector('.events-list');
    if (!list || !this.tycoon.state) return;

    const events = this.tycoon.state.events || [];
    if (events.length === 0) {
      list.innerHTML = '<div class="event-empty">No events yet. Start playing!</div>';
      return;
    }

    // Show last 10 events
    const recent = events.slice(-10).reverse();
    list.innerHTML = recent.map(e => `
      <div class="event-item event-${e.severity || 'info'}">
        <span class="event-tick">T${e.tick}</span>
        <span class="event-msg">${e.message}</span>
      </div>
    `).join('');
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
    this._renderPlanetPanel();
    this.refs.planetPanel?.classList.remove('hidden');
  }

  deselectPlanet() {
    this.selectedPlanetId = null;
    this.refs.planetPanel?.classList.add('hidden');
  }

  _renderPlanetPanel() {
    const panel = this.refs.planetPanel;
    if (!panel || !this.tycoon.state) return;

    const planet = this.tycoon.state.planets.find(p => p.id === this.selectedPlanetId);
    if (!planet) {
      panel.classList.add('hidden');
      return;
    }

    panel.classList.remove('hidden');

    const metaEl = panel.querySelector('.planet-meta');
    const colonyEl = panel.querySelector('.planet-colony');
    const buildingsEl = panel.querySelector('.planet-buildings');
    const actionsEl = panel.querySelector('.planet-actions');
    const nameEl = panel.querySelector('.panel-planet-name');

    nameEl.innerHTML = `${planet.type.emoji} ${planet.name}`;

    // Meta info
    metaEl.innerHTML = `
      <div class="meta-row"><span>Type</span><span>${planet.type.label}</span></div>
      <div class="meta-row"><span>Habitability</span><span class="stat-bar"><span class="stat-fill" style="width:${planet.stats.habitability}%"></span>${planet.stats.habitability}%</span></div>
      <div class="meta-row"><span>Stability</span><span class="stat-bar"><span class="stat-fill${planet.stats.stability < 30 ? ' danger' : planet.stats.stability < 60 ? ' warning' : ''}" style="width:${planet.stats.stability}%"></span>${planet.stats.stability}%</span></div>
      <div class="meta-row"><span>Mass</span><span>${planet.mass.toFixed(4)} M⊕</span></div>
      <div class="meta-row"><span>Primary</span><span>${RESOURCE_ICONS[planet.stats.primaryResource]} ${RESOURCE_LABELS[planet.stats.primaryResource]}</span></div>
      <div class="meta-row"><span>Secondary</span><span>${RESOURCE_ICONS[planet.stats.secondaryResource]} ${RESOURCE_LABELS[planet.stats.secondaryResource]}</span></div>
    `;

    if (!planet.colony) {
      // Uncolonized
      const costHtml = Object.entries(COLONIZATION_COST)
        .map(([res, amt]) => `${RESOURCE_ICONS[res]}${amt}`)
        .join(' ');
      const canAfford = this.tycoon.canAfford(COLONIZATION_COST);

      colonyEl.innerHTML = `<div class="colony-status">Uninhabited</div>`;
      buildingsEl.innerHTML = '';
      actionsEl.innerHTML = `
        <button class="tycoon-btn primary colonize-btn" data-planet="${planet.id}" ${!canAfford ? 'disabled' : ''}>
          🚀 Colonize — ${costHtml}
        </button>
      `;

      const btn = actionsEl.querySelector('.colonize-btn');
      if (btn && canAfford) {
        btn.addEventListener('click', () => {
          const result = this.tycoon.colonizePlanet(planet.id);
          if (!result.success) this._toast(result.error);
        });
      }
      return;
    }

    // Colonized
    const colony = planet.colony;
    colonyEl.innerHTML = `
      <div class="colony-status">🏛️ Colony Lv.${colony.level}</div>
      <div class="colony-pop">👥 Population: ${colony.population}</div>
      <div class="colony-founded">Founded: T${colony.foundedTick}</div>
    `;

    // Buildings
    if (colony.buildings.length > 0) {
      buildingsEl.innerHTML = '<div class="buildings-label">Buildings:</div>' +
        colony.buildings.map(bId => {
          const b = BUILDINGS.find(x => x.id === bId);
          return `<div class="building-chip">${b?.emoji || ''} ${b?.label || bId}</div>`;
        }).join('');
    } else {
      buildingsEl.innerHTML = '<div class="buildings-label empty">No buildings yet</div>';
    }

    // Build buttons
    const availableBuildings = BUILDINGS.filter(b => {
      if (b.id === 'gas_harvester' && planet.type.id !== 'gas_giant') return false;
      if (b.id === 'trade_hub' && colony.buildings.includes('trade_hub')) return false;
      return !colony.buildings.includes(b.id);
    });

    let buildHtml = '';
    for (const b of availableBuildings) {
      const costHtml = Object.entries(b.cost).map(([res, amt]) => `${RESOURCE_ICONS[res]}${amt}`).join(' ');
      const canAfford = this.tycoon.canAfford(b.cost);
      buildHtml += `<button class="tycoon-btn build-btn" data-planet="${planet.id}" data-building="${b.id}" ${!canAfford ? 'disabled' : ''}>
        ${b.emoji} ${b.label} — ${costHtml}
      </button>`;
    }

    // Trade button
    const otherColonies = this.tycoon.state.planets.filter(p => p.id !== planet.id && p.colony);
    let tradeHtml = '';
    if (colony.buildings.includes('trade_hub') && otherColonies.length > 0) {
      tradeHtml = '<div class="trade-label">Send trade ship:</div>';
      for (const target of otherColonies) {
        tradeHtml += `<button class="tycoon-btn trade-btn" data-from="${planet.id}" data-to="${target.id}">
          🚀 → ${target.name}
        </button>`;
      }
    }

    actionsEl.innerHTML = `
      <div class="build-section">${buildHtml || '<div class="no-builds">All buildings constructed</div>'}</div>
      ${tradeHtml}
    `;

    // Wire build buttons
    actionsEl.querySelectorAll('.build-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const bId = btn.dataset.building;
        const result = this.tycoon.buildOnPlanet(planet.id, bId);
        if (!result.success) this._toast(result.error);
      });
    });

    // Wire trade buttons
    actionsEl.querySelectorAll('.trade-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const fromId = parseInt(btn.dataset.from);
        const toId = parseInt(btn.dataset.to);
        const result = this.tycoon.sendTradeShip(fromId, toId);
        if (!result.success) this._toast(result.error);
      });
    });
  }

  // ── Event Handlers ─────────────────────────────────────────

  _onColonize(d) {
    this._toast(`🚀 Colonized ${d.planet.name}!`);
    this._renderPlanetPanel();
    this._renderResources();
    this._renderScore();
  }

  _onBuild(d) {
    this._toast(`🏗️ Built ${d.building.label} on ${d.planet.name}!`);
    this._renderPlanetPanel();
    this._renderResources();
  }

  _onTech(d) {
    this._toast(`🔬 Unlocked ${d.tech.label}!`);
    this._renderTech();
    this._renderResources();
    this._renderScore();
  }

  _onEvent(d) {
    this._renderEvents();
    this._renderResources();
    this._renderScore();
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
