const fs = require('fs');
let file = 'C:/Users/TX/Documents/Coding/projects/BrainDance/3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue';
let text = fs.readFileSync(file, 'utf8');

let newText = text.replace(/if\s*\(initialPoseMatrix\)\s*\{\s*console\.log\("\[Viewer\] Jumping to initial RAG pose"\);\s*flyToImage\(\{ matrix: initialPoseMatrix \}\);\s*\}/, '');

newText = newText.replace(/(\/\/ 最后调整相机，因为现在我们已经有了准确的 Center 和 Radius\s*adjustControlsToModel\(\);)/, '\\n          if (initialPoseMatrix) {\n            setTimeout(() => { flyToImage({ matrix: initialPoseMatrix }); }, 50);\n          }');

fs.writeFileSync(file, newText, 'utf8');
