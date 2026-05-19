const shirtPath = document.getElementById('shirt-path');
const shirtTextOverlay = document.getElementById('shirt-text-overlay');
const shirtColorInput = document.getElementById('shirt-color');
const textColorInput = document.getElementById('text-color');
const shirtTextInput = document.getElementById('shirt-text');
const sizeBtns = document.querySelectorAll('.size-btn');
const addToCartBtn = document.getElementById('add-to-cart');
const toastEl = document.getElementById('toast');

let selectedSize = 'L';

function updateShirt() {
    shirtPath.setAttribute('fill', shirtColorInput.value);
    shirtTextOverlay.style.color = textColorInput.value;
    shirtTextOverlay.textContent = shirtTextInput.value.trim().toUpperCase() || 'YOUR LOGO';
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

    if (navigator.clipboard) {
        navigator.clipboard.writeText(cartUrl).then(() => {
            toast('📋 Cart URL copied! Paste it on ULM to test.');
        }).catch(() => {
            toast('📋 ' + cartUrl);
        });
    } else {
        toast('📋 ' + cartUrl);
    }
});

updateShirt();
