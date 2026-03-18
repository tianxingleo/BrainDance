const fs = require('fs');
let code = fs.readFileSync('app/assets/webgl/assets/index-BIPQ5ZlA.js', 'utf8');

const s = 'clientX';
let idx = code.indexOf(s);
console.log('idx of clientX', idx);

if (idx > -1) {
    console.log('Context clientX:');
    console.log(code.substring(idx - 100, idx + 200));
}
