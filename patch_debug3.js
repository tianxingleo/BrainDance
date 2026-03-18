const fs = require('fs');
let code = fs.readFileSync('app/assets/webgl/assets/index-BIPQ5ZlA.js', 'utf8');

const s = code.substring(code.length - 2000);
const m = s.match(/k=Bn/);
if (m) {
    const idx = m.index;
    console.log(s.substring(idx - 10, idx + 50));
    console.log(Buffer.from(s.substring(idx, idx + 20)).toString('hex'));
} else {
    console.log('pattern not found at the end of file!');
}
