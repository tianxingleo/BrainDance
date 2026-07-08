const fs = require('fs');
let file = 'C:/Users/TX/Documents/Coding/projects/BrainDance/3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue';
let text = fs.readFileSync(file, 'utf8');

text = text.replace(
/await viewer\.addSplatScene\(currentPlyUrl,\s*\{\s*'showLoadingUI': true,\s*'progressiveLoad': true,\s*'splatAlphaTransferThreshold': 1,\s*'rotation': \[0, 0, 0, 1\],\s*\/\/\s*\[x, y, z, w\] Identity Quaternion \(No global rotation\)\s*\}\);\s*if\s*\(initialPoseMatrix\)\s*\{\s*console\.log\("\[Viewer\] Jumping to initial RAG pose"\);\s*flyToImage\(\{ matrix: initialPoseMatrix \}\);\s*\}/gs,
\wait viewer.addSplatScene(currentPlyUrl, {
        'showLoadingUI': true,
        'progressiveLoad': false,
        'rotation': [0, 0, 0, 1] // [x, y, z, w] Identity Quaternion (No global rotation)
      });\
);

text = text.replace(
/(\/\/ 最后调整相机，因为现在我们已经有了准确的 Center 和 Radius\s*adjustControlsToModel\(\);)/gs,
\\

          // 随后再直接触发视角平移
          if (initialPoseMatrix) {
            console.log("[Viewer] Jumping to initial RAG pose");
            setTimeout(() => {
              flyToImage({ matrix: initialPoseMatrix });
            }, 50);
          }\
);

fs.writeFileSync(file, text, 'utf8');
console.log('DONE!');
