const fs = require('fs');
let code = fs.readFileSync('app/assets/webgl/assets/index-BIPQ5ZlA.js', 'utf8');

const s1 = 'W=X=>{k.value=!0';
let idx1 = code.indexOf(s1);
console.log('idx1', idx1);

if (idx1 > -1) {
    console.log('Context W:');
    console.log(code.substring(idx1 - 50, idx1 + 200));
}

const s2 = 'window.addEventListener("mousedown",W)';
let idx2 = code.indexOf(s2);
console.log('idx2', idx2);

if (idx2 > -1) {
    console.log('Context addEventListener:');
    console.log(code.substring(idx2 - 50, idx2 + 200));
}

const s3 = 'onMousedown:W';
let idx3 = code.indexOf(s3);
console.log('idx3', idx3);

if (idx3 > -1) {
    console.log('Context onMousedown:');
    console.log(code.substring(idx3 - 50, idx3 + 200));
}
