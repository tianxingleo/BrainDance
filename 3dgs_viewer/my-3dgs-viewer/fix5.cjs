const fs = require('fs');
let file = 'C:/Users/TX/Documents/Coding/projects/BrainDance/3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue';
let text = fs.readFileSync(file, 'utf8');

text = text.replaceAll('\\n', '\n');

fs.writeFileSync(file, text, 'utf8');
