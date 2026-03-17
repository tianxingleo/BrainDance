const fs = require('fs');
let file = 'C:/Users/TX/Documents/Coding/projects/BrainDance/3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue';
let text = fs.readFileSync(file, 'utf8');

text = text.replace(/        'rotation': \[0, 0, 0, 1\] \/\/ \[x, y, z, w\] Identity Quaternion \(No global rotation\)\r?\n\r?\n\r?\n      \/\/ 告诉/, "        'rotation': [0, 0, 0, 1] // [x, y, z, w] Identity Quaternion (No global rotation)\n      });\n\n      // 告诉");

fs.writeFileSync(file, text, 'utf8');
