const fs = require('fs');
let code = fs.readFileSync('app/assets/webgl/assets/index-BIPQ5ZlA.js', 'utf8');

let originalCode = code;

code = code.replace(
    /k=Bn\(!1\),z=\{x:0,y:0\},W=X=>\{k\.value=!0,z\.x=X\.clientX,z\.y=X\.clientY\},H=X=>\{if\(!k\.value\|\|!g\|\|!g\.camera\)return;const Z=X\.clientX-z\.x,pe=X\.clientY-z\.y,ge=\.2,be=-Z\*ge,Me=-pe\*ge;x\.value\.x\+=Me,x\.value\.y\+=be,g\.camera\.rotateX\(Me\*Math\.PI\/180\),g\.camera\.rotateOnWorldAxis\(new L\(0,1,0\),be\*Math\.PI\/180\),g\.camera\.updateProjectionMatrix\(\),m\(\),z\.x=X\.clientX,z\.y=X\.clientY\}/g,
    'k=Bn(!1),z={x:0,y:0},W=X=>{k.value=!0;let cx=X.touches?X.touches[0].clientX:X.clientX,cy=X.touches?X.touches[0].clientY:X.clientY;z.x=cx,z.y=cy},H=X=>{if(!k.value||!g||!g.camera)return;X.touches&&X.preventDefault&&X.preventDefault();let cx=X.touches?X.touches[0].clientX:X.clientX,cy=X.touches?X.touches[0].clientY:X.clientY;const Z=cx-z.x,pe=cy-z.y,ge=.2,be=-Z*ge,Me=-pe*ge;x.value.x+=Me,x.value.y+=be,g.camera.rotateX(Me*Math.PI/180),g.camera.rotateOnWorldAxis(new L(0,1,0),be*Math.PI/180),g.camera.updateProjectionMatrix(),m(),z.x=cx,z.y=cy}'
);

code = code.replace(
    /window\.addEventListener\("mousedown",W\),window\.addEventListener\("mousemove",H\),window\.addEventListener\("mouseup",\$\)/g,
    'window.addEventListener("mousedown",W),window.addEventListener("mousemove",H),window.addEventListener("mouseup",$),window.addEventListener("touchstart",W,{passive:!1}),window.addEventListener("touchmove",H,{passive:!1}),window.addEventListener("touchend",$),window.addEventListener("touchcancel",$)'
);

code = code.replace(
    /window\.removeEventListener\("mousemove",H\),window\.removeEventListener\("mouseup",\$\)/g,
    'window.removeEventListener("mousemove",H),window.removeEventListener("mouseup",$),window.removeEventListener("touchmove",H),window.removeEventListener("touchend",$),window.removeEventListener("touchcancel",$)'
);

code = code.replace(
    /onMousedown:W,onMousemove:H,onMouseup:\$,onMouseleave:\$/g,
    'onMousedown:W,onMousemove:H,onMouseup:$,onMouseleave:$,onTouchstart:W,onTouchmove:H,onTouchend:$,onTouchcancel:$'
);

if (code === originalCode) {
    console.error("No replacements made!");
    process.exit(1);
} else {
    fs.writeFileSync('app/assets/webgl/assets/index-BIPQ5ZlA.js', code);
    console.log('Patch complete. Repalacements successful.');
}
