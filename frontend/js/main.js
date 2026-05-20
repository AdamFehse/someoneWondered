const container = document.getElementById('three-canvas');
const shirtColorInput = document.getElementById('shirt-color');
const textColorInput = document.getElementById('text-color');
const shirtTextInput = document.getElementById('shirt-text');
const sizeBtns = document.querySelectorAll('.size-btn');
const addToCartBtn = document.getElementById('add-to-cart');
const toastEl = document.getElementById('toast');

let selectedSize = 'L';
let scene, camera, renderer;
let shirtGroup, shirtMesh, particleSystem, confettiSystem;
let isDragging = false, prevMouse = { x: 0, y: 0 };
let isCelebrating = false, celebrationTimer = 0;
let toastTimeout;

function initScene() {
  scene = new THREE.Scene();

  const w = container.clientWidth, h = container.clientHeight;
  camera = new THREE.PerspectiveCamera(35, w / h, 1, 1000);
  camera.position.set(0, 12, 220);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.5));

  const key = new THREE.DirectionalLight(0xffffff, 0.7);
  key.position.set(1, 2.5, 2);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0x8888ff, 0.3);
  fill.position.set(-1.2, 1, 1.5);
  scene.add(fill);

  const rim = new THREE.DirectionalLight(0xffffff, 0.35);
  rim.position.set(0, -1, -1.5);
  scene.add(rim);
}

function createShirtShape() {
  const s = new THREE.Shape();
  s.moveTo(55, 280);
  s.lineTo(55, 130);
  s.quadraticCurveTo(25, 130, 15, 110);
  s.quadraticCurveTo(5, 85, 20, 65);
  s.quadraticCurveTo(30, 50, 50, 47);
  s.quadraticCurveTo(65, 44, 76, 44);
  s.quadraticCurveTo(85, 28, 95, 28);
  s.quadraticCurveTo(105, 28, 114, 44);
  s.quadraticCurveTo(125, 44, 140, 47);
  s.quadraticCurveTo(160, 50, 170, 65);
  s.quadraticCurveTo(185, 85, 175, 110);
  s.quadraticCurveTo(165, 130, 135, 130);
  s.lineTo(135, 280);
  s.quadraticCurveTo(95, 290, 55, 280);
  return s;
}

function buildShirt() {
  const shape = createShirtShape();
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: 12,
    bevelEnabled: true,
    bevelSegments: 3,
    bevelSize: 1.5,
    bevelThickness: 2.5,
  });
  geo.center();

  shirtMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    roughness: 0.85,
    metalness: 0.0,
  }));
  shirtMesh.castShadow = true;

  shirtGroup = new THREE.Group();
  shirtGroup.add(shirtMesh);
  scene.add(shirtGroup);

  updateTexture();
}

function updateTexture() {
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = 1024;
  const ctx = c.getContext('2d');

  ctx.fillStyle = shirtColorInput.value;
  ctx.fillRect(0, 0, c.width, c.height);

  ctx.globalAlpha = 0.04;
  for (let i = 0; i < 3000; i++) {
    ctx.fillStyle = Math.random() > 0.5 ? '#fff' : '#000';
    ctx.fillRect(Math.random() * c.width, Math.random() * c.height, 1, 1);
  }
  ctx.globalAlpha = 1;

  const text = shirtTextInput.value.trim().toUpperCase() || 'YOUR LOGO';
  const fs = text.length > 8 ? 110 : text.length > 4 ? 150 : 190;

  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  ctx.shadowColor = 'rgba(0,0,0,0.25)';
  ctx.shadowBlur = 10;
  ctx.shadowOffsetX = 3;
  ctx.shadowOffsetY = 3;
  ctx.font = `900 ${fs}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
  ctx.fillStyle = textColorInput.value;
  ctx.fillText(text, c.width / 2, c.height / 2 + 20);
  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  ctx.fillText(text, c.width / 2, c.height / 2 + 20);

  if (!shirtMesh) return;
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  if (shirtMesh.material.map) shirtMesh.material.map.dispose();
  shirtMesh.material.map = tex;
  shirtMesh.material.needsUpdate = true;
}

function createParticles() {
  const count = 120;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const t = Math.random() * Math.PI * 2;
    const p = Math.acos(2 * Math.random() - 1);
    const r = 90 + Math.random() * 110;
    pos[i * 3] = r * Math.sin(p) * Math.cos(t);
    pos[i * 3 + 1] = (Math.random() - 0.5) * 280;
    pos[i * 3 + 2] = r * Math.sin(p) * Math.sin(t);
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  const mat = new THREE.PointsMaterial({
    size: 2.5,
    color: 0xffffff,
    transparent: true,
    opacity: 0.35,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });
  particleSystem = new THREE.Points(geo, mat);
  scene.add(particleSystem);
}

function spawnConfetti() {
  const count = 250;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  const col = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 20;
    pos[i * 3 + 1] = (Math.random() - 0.5) * 20;
    pos[i * 3 + 2] = (Math.random() - 0.5) * 20;
    const c = new THREE.Color().setHSL(Math.random(), 0.85, 0.55 + Math.random() * 0.2);
    col[i * 3] = c.r;
    col[i * 3 + 1] = c.g;
    col[i * 3 + 2] = c.b;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));

  confettiSystem = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 5,
    vertexColors: true,
    transparent: true,
    opacity: 1,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  }));
  scene.add(confettiSystem);
}

function setupDrag() {
  const el = renderer.domElement;
  el.addEventListener('mousedown', e => {
    isDragging = true;
    prevMouse = { x: e.clientX, y: e.clientY };
  });
  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = e.clientX - prevMouse.x;
    const dy = e.clientY - prevMouse.y;
    shirtGroup.rotation.y += dx * 0.008;
    shirtGroup.rotation.x += dy * 0.008;
    prevMouse = { x: e.clientX, y: e.clientY };
  });
  window.addEventListener('mouseup', () => { isDragging = false; });

  el.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      isDragging = true;
      prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
  }, { passive: true });
  el.addEventListener('touchmove', e => {
    if (!isDragging || e.touches.length !== 1) return;
    const dx = e.touches[0].clientX - prevMouse.x;
    const dy = e.touches[0].clientY - prevMouse.y;
    shirtGroup.rotation.y += dx * 0.008;
    shirtGroup.rotation.x += dy * 0.008;
    prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
  }, { passive: true });
  el.addEventListener('touchend', () => { isDragging = false; });
}

function animate() {
  requestAnimationFrame(animate);

  if (shirtGroup && !isDragging) {
    shirtGroup.rotation.y += 0.004;
    shirtGroup.position.y = Math.sin(Date.now() * 0.0012) * 1.5;
  }

  if (isCelebrating && shirtGroup) {
    celebrationTimer += 0.02;
    shirtGroup.rotation.y += 0.18;
    shirtGroup.rotation.x = Math.sin(celebrationTimer * 4) * 0.35;
    const p = 1 + Math.sin(celebrationTimer * 10) * 0.03;
    shirtGroup.scale.setScalar(p);
    if (celebrationTimer > 2.5) {
      isCelebrating = false;
      shirtGroup.rotation.x = 0;
      shirtGroup.scale.setScalar(1);
    }
  }

  if (particleSystem) {
    const p = particleSystem.geometry.attributes.position.array;
    for (let i = 0; i < p.length / 3; i++) {
      p[i * 3 + 1] += 0.1;
      if (p[i * 3 + 1] > 160) p[i * 3 + 1] = -160;
    }
    particleSystem.geometry.attributes.position.needsUpdate = true;
  }

  if (confettiSystem) {
    const p = confettiSystem.geometry.attributes.position.array;
    const d = confettiSystem.userData || (confettiSystem.userData = { elapsed: 0 });
    d.elapsed += 0.02;
    for (let i = 0; i < p.length / 3; i++) {
      p[i * 3] += (Math.random() - 0.5) * 0.6;
      p[i * 3 + 1] += 0.35 - 0.035 * d.elapsed;
      p[i * 3 + 2] += (Math.random() - 0.5) * 0.6;
    }
    confettiSystem.geometry.attributes.position.needsUpdate = true;
    confettiSystem.material.opacity = Math.max(0, 1 - d.elapsed / 3);
    if (d.elapsed > 3) {
      scene.remove(confettiSystem);
      confettiSystem.geometry.dispose();
      confettiSystem.material.dispose();
      confettiSystem = null;
    }
  }

  renderer.render(scene, camera);
}

function toast(msg, duration = 2500) {
  toastEl.textContent = msg;
  toastEl.classList.remove('hidden');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toastEl.classList.add('hidden'), duration);
}

function handleAddToCart() {
  const text = shirtTextInput.value.trim();
  const params = new URLSearchParams({
    'add-to-cart': 123,
    'quantity': 1,
    'merch_text': text || 'ULM',
    'merch_shirt_color': shirtColorInput.value,
    'merch_text_color': textColorInput.value,
    'merch_size': selectedSize,
  });
  const cartUrl = `https://ulmproductions.org/cart/?${params.toString()}`;

  isCelebrating = true;
  celebrationTimer = 0;
  spawnConfetti();

  if (navigator.clipboard) {
    navigator.clipboard.writeText(cartUrl).then(() => {
      toast('✨ cart url copied! (it\'s for fun)');
    }).catch(() => {
      toast('✨ ' + cartUrl);
    });
  } else {
    toast('✨ ' + cartUrl);
  }
}

function init() {
  if (typeof THREE === 'undefined') {
    container.innerHTML = '<p style="padding:2rem;color:var(--text-muted)">3D renderer unavailable</p>';
    return;
  }
  initScene();
  buildShirt();
  createParticles();
  setupDrag();
  animate();

  shirtColorInput.addEventListener('input', updateTexture);
  textColorInput.addEventListener('input', updateTexture);
  shirtTextInput.addEventListener('input', updateTexture);

  sizeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sizeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedSize = btn.dataset.size;
    });
  });

  addToCartBtn.addEventListener('click', handleAddToCart);

  window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

init();
