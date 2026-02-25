const fs = require('fs');
let code = fs.readFileSync('app/assets/webgl/assets/index-BIPQ5ZlA.js', 'utf8');

// Replace W and H functions to support touch
code = code.replace(
    /W\s*=\s*X\s*=>\s*\{\s*k\.value\s*=\s*!0\s*,\s*z\.x\s*=\s*X\.clientX\s*,\s*z\.y\s*=\s*X\.clientY\s*\}\s*,\s*H\s*=\s*X\s*=>\s*\{\s*if\s*\(\!k\.value\s*\|\|\s*!g\s*\|\|\s*!g\.camera\)\s*return;\s*const\s*Z\s*=\s*X\.clientX\s*-\s*z\.x\s*,\s*pe\s*=\s*X\.clientY\s*-\s*z\.y/g,
    `W = X => { k.value = !0; let cx = X.touches ? X.touches[0].clientX : X.clientX, cy = X.touches ? X.touches[0].clientY : X.clientY; z.x = cx; z.y = cy; }, H = X => { if (!k.value || !g || !g.camera) return; X.touches && typeof X.preventDefault === 'function' && X.preventDefault(); let cx = X.touches ? X.touches[0].clientX : X.clientX, cy = X.touches ? X.touches[0].clientY : X.clientY; const Z = cx - z.x, pe = cy - z.y`
);

code = code.replace(
    /g\.camera\.updateProjectionMatrix\(\)\s*,\s*m\(\)\s*,\s*z\.x\s*=\s*X\.clientX\s*,\s*z\.y\s*=\s*X\.clientY/g,
    `g.camera.updateProjectionMatrix(), m(), z.x = (X.touches ? X.touches[0].clientX : X.clientX), z.y = (X.touches ? X.touches[0].clientY : X.clientY)`
);

// Replace addEventListener
code = code.replace(
    /window\.addEventListener\("mousedown"\s*,\s*W\)\s*,\s*window\.addEventListener\("mousemove"\s*,\s*H\)\s*,\s*window\.addEventListener\("mouseup"\s*,\s*\$\)/g,
    `window.addEventListener("mousedown", W), window.addEventListener("mousemove", H), window.addEventListener("mouseup", $), window.addEventListener("touchstart", W, {passive: !1}), window.addEventListener("touchmove", H, {passive: !1}), window.addEventListener("touchend", $), window.addEventListener("touchcancel", $)`
);

// Replace removeEventListener
code = code.replace(
    /window\.removeEventListener\("mousemove"\s*,\s*H\)\s*,\s*window\.removeEventListener\("mouseup"\s*,\s*\$\)/g,
    `window.removeEventListener("mousemove", H), window.removeEventListener("mouseup", $), window.removeEventListener("touchmove", H), window.removeEventListener("touchend", $), window.removeEventListener("touchcancel", $)`
);

// Replace component listeners
code = code.replace(
    /onMousedown:\s*W\s*,\s*onMousemove:\s*H\s*,\s*onMouseup:\s*\$\s*,\s*onMouseleave:\s*\$/g,
    `onMousedown: W, onMousemove: H, onMouseup: $, onMouseleave: $, onTouchstart: W, onTouchmove: H, onTouchend: $, onTouchcancel: $`
);

fs.writeFileSync('app/assets/webgl/assets/index-BIPQ5ZlA.js', code);
console.log('REPLACING COMPLETE.');
