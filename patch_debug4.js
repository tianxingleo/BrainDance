const fs = require('fs');
let code = fs.readFileSync('app/assets/webgl/assets/index-BIPQ5ZlA.js', 'utf8');

const m = code.match(/z=\{x:0,y:0\}/);
if (m) {
    const idx = m.index;
    console.log("MATCH 1: ", code.substring(idx - 10, idx + 50));
} else {
    console.log('pattern z={x:0,y:0} not found!');
}

const m2 = code.match(/\.clientX/g);
console.log('clientX matches:', m2 ? m2.length : 0);
