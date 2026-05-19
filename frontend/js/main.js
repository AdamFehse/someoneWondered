const canvas = document.getElementById('shirt-canvas');
const ctx = canvas.getContext('2d');

const shirtColorInput = document.getElementById('shirt-color');
const textColorInput = document.getElementById('text-color');
const shirtTextInput = document.getElementById('shirt-text');
const sizeBtns = document.querySelectorAll('.size-btn');
const addToCartBtn = document.getElementById('add-to-cart');
const toastEl = document.getElementById('toast');

let selectedSize = 'L';

const SHIRT_COLLAR_MODEL = 'ULM-2025';

function drawShirt() {
    const w = canvas.width;
    const h = canvas.height;
    const shirtColor = shirtColorInput.value;
    const textColor = textColorInput.value;
    const text = shirtTextInput.value.trim() || 'Your Logo';

    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.3)';
    ctx.shadowBlur = 30;
    ctx.shadowOffsetY = 8;

    ctx.beginPath();
    ctx.moveTo(w * 0.28, h * 0.05);
    ctx.quadraticCurveTo(w * 0.35, h * 0.01, w * 0.4, h * 0.05);
    ctx.lineTo(w * 0.4, h * 0.13);

    ctx.quadraticCurveTo(w * 0.32, h * 0.16, w * 0.08, h * 0.06);
    ctx.quadraticCurveTo(w * 0.04, h * 0.10, w * 0.10, h * 0.20);

    ctx.lineTo(w * 0.22, h * 0.94);
    ctx.quadraticCurveTo(w * 0.24, h * 0.99, w * 0.30, h);

    ctx.lineTo(w * 0.70, h);
    ctx.quadraticCurveTo(w * 0.76, h * 0.99, w * 0.78, h * 0.94);

    ctx.lineTo(w * 0.90, h * 0.20);
    ctx.quadraticCurveTo(w * 0.96, h * 0.10, w * 0.92, h * 0.06);
    ctx.quadraticCurveTo(w * 0.68, h * 0.16, w * 0.60, h * 0.13);
    ctx.lineTo(w * 0.60, h * 0.05);
    ctx.quadraticCurveTo(w * 0.65, h * 0.01, w * 0.72, h * 0.05);

    ctx.closePath();
    ctx.restore();

    ctx.fillStyle = shirtColor;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();

    const r = parseInt(shirtColor.slice(1,3), 16);
    const g = parseInt(shirtColor.slice(3,5), 16);
    const b = parseInt(shirtColor.slice(5,7), 16);
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

    const highlight = luminance > 0.5 ? 'rgba(0,0,0,0.04)' : 'rgba(255,255,255,0.06)';
    ctx.beginPath();
    ctx.moveTo(w * 0.35, h * 0.06);
    ctx.quadraticCurveTo(w * 0.45, h * 0.30, w * 0.40, h * 0.45);
    ctx.quadraticCurveTo(w * 0.35, h * 0.30, w * 0.28, h * 0.06);
    ctx.closePath();
    ctx.fillStyle = highlight;
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(w * 0.55, h * 0.50);
    ctx.quadraticCurveTo(w * 0.65, h * 0.65, w * 0.72, h * 0.50);
    ctx.quadraticCurveTo(w * 0.65, h * 0.55, w * 0.55, h * 0.50);
    ctx.closePath();
    ctx.fillStyle = highlight;
    ctx.fill();

    const fontSize = Math.min(w, h) * 0.06;
    ctx.fillStyle = textColor;
    ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const maxWidth = w * 0.55;
    let displayText = text;
    if (ctx.measureText(displayText).width > maxWidth) {
        while (ctx.measureText(displayText + '…').width > maxWidth && displayText.length > 1) {
            displayText = displayText.slice(0, -1);
        }
        displayText += '…';
    }

    const textY = h * 0.58;
    ctx.shadowColor = 'rgba(0,0,0,0.2)';
    ctx.shadowBlur = 4;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 1;
    ctx.fillText(displayText.toUpperCase(), w / 2, textY);
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    ctx.fillStyle = textColor;
    ctx.globalAlpha = 0.15;
    ctx.font = `bold ${fontSize * 0.3}px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
    ctx.fillText(SHIRT_COLLAR_MODEL, w / 2, h * 0.23);
    ctx.globalAlpha = 1;
}

function updateShirt() {
    drawShirt();
}

shirtColorInput.addEventListener('input', updateShirt);
textColorInput.addEventListener('input', updateShirt);
shirtTextInput.addEventListener('input', updateShirt);

sizeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        sizeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedSize = btn.dataset.size;
    });
});

let toastTimeout;

function toast(msg, duration = 2500) {
    toastEl.textContent = msg;
    toastEl.classList.remove('hidden');
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => toastEl.classList.add('hidden'), duration);
}

addToCartBtn.addEventListener('click', () => {
    const text = shirtTextInput.value.trim();
    const shirtColor = shirtColorInput.value;
    const textColorVal = textColorInput.value;
    const productId = 123;
    const baseUrl = 'https://ulmproductions.org/cart/';

    const params = new URLSearchParams({
        'add-to-cart': productId,
        'quantity': 1,
        'merch_text': text || 'ULM',
        'merch_shirt_color': shirtColor,
        'merch_text_color': textColorVal,
        'merch_size': selectedSize,
    });

    const cartUrl = `${baseUrl}?${params.toString()}`;

    navigator.clipboard.writeText(cartUrl).then(() => {
        toast('📋 Cart URL copied! Paste it to test the WooCommerce flow.');
    }).catch(() => {
        toast('📋 ' + cartUrl);
    });
});

updateShirt();
