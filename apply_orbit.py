"""Apply orbit mode additions to the flutter branch base file, with correct UTF-8 encoding."""

# Read the flutter branch base
with open('spark_flutter_base.vue', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# We'll build the output by inserting orbit code at specific locations
output = []
i = 0

while i < len(lines):
    line = lines[i]
    stripped = line.rstrip('\n')

    # === Insert orbit state variables after currentPosesPath declaration ===
    if stripped == "const currentPosesPath = ref('/models/webgl_poses_with_tags.json');":
        output.append(line)
        i += 1
        # Insert orbit variables
        output.append('\n')
        output.append('// ==================== Orbit 相机模式 ====================\n')
        output.append('// Orbit 模式：相机绕模型中心自动旋转（圆周运动），不改变现有手动控制逻辑\n')
        output.append("const orbitEnabled = ref(false);    // 是否开启 orbit 模式\n")
        output.append("const orbitPaused = ref(false);     // 是否暂停旋转\n")
        output.append("const orbitSpeed = ref(20);         // 旋转速度，单位：度/秒\n")
        output.append("const orbitDirection = ref(1);      // 旋转方向：1=逆时针(CCW)，-1=顺时针(CW)\n")
        output.append("const orbitRadius = ref(0);         // 旋转半径：0=自动计算（保持当前距离），>0 使用指定值\n")
        continue

    # === Insert orbit runtime variables after currentPoseIndex ===
    if stripped == "const currentPoseIndex = ref(-1);":
        output.append(line)
        i += 1
        output.append('\n')
        output.append('// ===== Orbit 运行时变量（非响应式，避免不必要的 Vue 重渲染） =====\n')
        output.append("let orbitAngle = 0;           // 当前旋转角度（弧度）\n")
        output.append("let orbitY = 0;               // 相机 Y 坐标（保持 orbit 高度不变）\n")
        output.append("let autoOrbitRadius = 0;      // 自动计算的轨道半径（从当前相机距离获取）\n")
        output.append("let orbitLastFrameTime = performance.now();\n")
        output.append("let isOrbitDragging = false;  // 用户正在手动拖拽（临时中断 orbit）\n")
        continue

    # === Insert orbit functions after syncClipPlane function ===
    if stripped == '};' and i > 0 and 'updateClipPlaneEffect' in lines[i-1]:
        output.append(line)
        i += 1
        output.append('\n')
        output.append('// ==================== Orbit 相机模式：核心函数 ====================\n')
        output.append('\n')
        output.append('// 从当前相机位置同步 orbit 参数（角度、高度、半径）\n')
        output.append('// 用于：(1) 开启 orbit 时初始化 (2) 手动拖拽松开后恢复 (3) 模型加载后重新锚定\n')
        output.append('const syncOrbitFromCamera = () => {\n')
        output.append('  if (!camera || !sceneCenter) return;\n')
        output.append('  const dx = camera.position.x - sceneCenter.x;\n')
        output.append('  const dz = camera.position.z - sceneCenter.z;\n')
        output.append('  orbitAngle = Math.atan2(dz, dx);\n')
        output.append('  orbitY = camera.position.y;\n')
        output.append('  autoOrbitRadius = Math.sqrt(dx * dx + dz * dz);\n')
        output.append('};\n')
        output.append('\n')
        output.append('// 开启 orbit 模式：从当前位置初始化轨道参数并开始旋转\n')
        output.append('const startOrbit = () => {\n')
        output.append('  if (!sceneCenter) return;\n')
        output.append('  syncOrbitFromCamera();\n')
        output.append('  // 如果自动半径过小（相机太靠近中心），使用默认距离\n')
        output.append('  if (autoOrbitRadius < 0.1) {\n')
        output.append('    autoOrbitRadius = sceneRadius * 2.4;\n')
        output.append('  }\n')
        output.append('  orbitPaused.value = false;\n')
        output.append('  orbitEnabled.value = true;\n')
        output.append('};\n')
        output.append('\n')
        output.append('// 关闭 orbit 模式：相机停在当前位置，恢复原有手动控制\n')
        output.append('const stopOrbit = () => {\n')
        output.append('  orbitEnabled.value = false;\n')
        output.append('  orbitPaused.value = false;\n')
        output.append('};\n')
        output.append('\n')
        output.append('// 暂停/恢复 orbit 旋转\n')
        output.append('const toggleOrbitPause = () => {\n')
        output.append('  if (!orbitEnabled.value) return;\n')
        output.append('  if (orbitPaused.value) {\n')
        output.append('    // 恢复时从当前位置重新同步轨道参数，确保无缝衔接\n')
        output.append('    syncOrbitFromCamera();\n')
        output.append('    orbitPaused.value = false;\n')
        output.append('  } else {\n')
        output.append('    orbitPaused.value = true;\n')
        output.append('  }\n')
        output.append('};\n')
        output.append('\n')
        output.append('// 切换旋转方向（顺时针/逆时针）\n')
        output.append('const toggleOrbitDirection = () => {\n')
        output.append('  orbitDirection.value *= -1;\n')
        output.append('};\n')
        continue

    # === Insert pointer event listeners after renderer.domElement is appended ===
    if stripped == 'containerRef.value.appendChild(renderer.domElement);':
        output.append(line)
        i += 1
        output.append('\n')
        output.append('    // ===== Orbit 指针事件：检测手动拖拽以临时中断/恢复 orbit 旋转 =====\n')
        output.append("    renderer.domElement.addEventListener('pointerdown', () => {\n")
        output.append('      isOrbitDragging = true;\n')
        output.append('    });\n')
        output.append("    renderer.domElement.addEventListener('pointerup', () => {\n")
        output.append('      isOrbitDragging = false;\n')
        output.append('      if (orbitEnabled.value && !orbitPaused.value) {\n')
        output.append('        syncOrbitFromCamera();\n')
        output.append('      }\n')
        output.append('    });\n')
        output.append("    renderer.domElement.addEventListener('pointerleave', () => {\n")
        output.append('      if (isOrbitDragging) {\n')
        output.append('        isOrbitDragging = false;\n')
        output.append('        if (orbitEnabled.value && !orbitPaused.value) {\n')
        output.append('          syncOrbitFromCamera();\n')
        output.append('        }\n')
        output.append('      }\n')
        output.append('    });\n')
        continue

    # === Replace animation loop with orbit-enhanced version ===
    if stripped == '    renderer.setAnimationLoop(() => {':
        # Skip the original animation loop until the closing });
        # First, write the new animation loop header
        output.append(line)
        i += 1
        # Copy lines until "renderer.render(scene, camera);"
        while i < len(lines):
            l = lines[i]
            s = l.rstrip('\n')
            if s.strip() == 'renderer.render(scene, camera);':
                # Insert orbit logic before render
                output.append('\n')
                output.append('      const now = performance.now();\n')
                output.append('      const orbitDt = Math.min((now - orbitLastFrameTime) / 1000, 0.1);\n')
                output.append('      orbitLastFrameTime = now;\n')
                output.append('\n')
                output.append('      // FPS 统计\n')
                output.append('      fpsFrames += 1;\n')
                output.append('      if (now - fpsTimestamp >= 1000) {\n')
                output.append('        currentFps.value = fpsFrames;\n')
                output.append('        fpsFrames = 0;\n')
                output.append('        fpsTimestamp = now;\n')
                output.append('      }\n')
                output.append('\n')
                output.append('      // ===== Orbit 模式：自动旋转相机（非暂停、非手动拖拽时生效） =====\n')
                output.append('      if (orbitEnabled.value && !orbitPaused.value && !isOrbitDragging && orbitDt > 0) {\n')
                output.append('        const speedRad = orbitSpeed.value * (Math.PI / 180);\n')
                output.append('        orbitAngle += speedRad * orbitDt * orbitDirection.value;\n')
                output.append('        const r = orbitRadius.value > 0 ? orbitRadius.value : autoOrbitRadius;\n')
                output.append('        const x = sceneCenter.x + r * Math.cos(orbitAngle);\n')
                output.append('        const z = sceneCenter.z + r * Math.sin(orbitAngle);\n')
                output.append('        camera.position.set(x, orbitY, z);\n')
                output.append('        camera.lookAt(sceneCenter);\n')
                output.append('        camera.updateProjectionMatrix();\n')
                output.append('      }\n')
                output.append('\n')
                output.append('      // ===== Orbit 模式下重新锁定相机位置，防止 cameraRig 覆盖 orbit 位置 =====\n')
                output.append('      if (orbitEnabled.value && !orbitPaused.value && !isOrbitDragging) {\n')
                output.append('        const r = orbitRadius.value > 0 ? orbitRadius.value : autoOrbitRadius;\n')
                output.append('        const x = sceneCenter.x + r * Math.cos(orbitAngle);\n')
                output.append('        const z = sceneCenter.z + r * Math.sin(orbitAngle);\n')
                output.append('        camera.position.set(x, orbitY, z);\n')
                output.append('        camera.lookAt(sceneCenter);\n')
                output.append('      }\n')
                output.append('\n')
                output.append('      renderer.render(scene, camera);\n')
                i += 1
                # Skip the original FPS tracking block (fpsFrames += 1 through the closing brace)
                skip_fps = True
                while i < len(lines):
                    s2 = lines[i].rstrip('\n').strip()
                    if skip_fps and (s2.startswith('fpsFrames += 1') or
                                     s2.startswith('const now = performance.now()') or
                                     s2.startswith('if (now - fpsTimestamp') or
                                     s2.startswith('currentFps.value = fpsFrames') or
                                     s2.startswith('fpsFrames = 0') or
                                     s2.startswith('fpsTimestamp = now') or
                                     s2 == '}' or
                                     s2 == ''):
                        i += 1
                        if s2 == '}':
                            break
                        continue
                    break
                break
            else:
                output.append(l)
                i += 1
        continue

    # === Insert orbit sync after sceneCenter calculation ===
    if stripped == 'sceneRadius = Math.max(size.length() * 0.32, DEFAULT_SCENE_RADIUS);':
        output.append(line)
        i += 1
        output.append('\n')
        output.append('    // 如果 orbit 已开启，重新同步轨道参数到新计算的模型中心\n')
        output.append('    if (orbitEnabled.value) {\n')
        output.append('      syncOrbitFromCamera();\n')
        output.append('    }\n')
        continue

    # === Insert orbit panel before closing </div></template> ===
    if stripped == '    </div>' and i + 1 < len(lines) and lines[i+1].rstrip('\n') == '</template>':
        output.append(line)
        i += 1
        output.append('\n')
        output.append('    <!-- ===== Orbit 控制面板：相机绕模型中心自动旋转 ===== -->\n')
        output.append('    <div\n')
        output.append('      v-if="!isLoading && !loadError"\n')
        output.append('      class="orbit-panel panel-card"\n')
        output.append('      @mousedown.stop\n')
        output.append('      @touchstart.stop\n')
        output.append('      @touchmove.stop\n')
        output.append('      @touchend.stop\n')
        output.append('      @touchcancel.stop\n')
        output.append('    >\n')
        output.append('      <div class="eyebrow">Orbit Control</div>\n')
        output.append('      <div class="panel-title">轨道旋转</div>\n')
        output.append('      <div class="orbit-btn-row">\n')
        output.append('        <button class="panel-btn panel-btn--solid" @click="orbitEnabled ? stopOrbit() : startOrbit()">\n')
        output.append("          {{ orbitEnabled ? '关闭轨道' : '开启轨道' }}\n")
        output.append('        </button>\n')
        output.append('        <button v-if="orbitEnabled" class="panel-btn panel-btn--ghost" @click="toggleOrbitPause()">\n')
        output.append("          {{ orbitPaused ? '恢复' : '暂停' }}\n")
        output.append('        </button>\n')
        output.append('      </div>\n')
        output.append('      <template v-if="orbitEnabled">\n')
        output.append('        <!-- 旋转速度控制 -->\n')
        output.append('        <div class="focal-row" style="margin-top: 10px;">\n')
        output.append('          <span>速度</span>\n')
        output.append("          <span>{{ orbitSpeed }} 度/秒</span>\n")
        output.append('        </div>\n')
        output.append('        <input\n')
        output.append('          type="range"\n')
        output.append('          :min="1"\n')
        output.append('          :max="120"\n')
        output.append('          :value="orbitSpeed"\n')
        output.append('          step="1"\n')
        output.append('          @input="orbitSpeed = Number($event.target.value)"\n')
        output.append('        />\n')
        output.append('        <!-- 旋转半径控制 -->\n')
        output.append('        <div class="focal-row" style="margin-top: 6px;">\n')
        output.append('          <span>半径</span>\n')
        output.append("          <span>{{ orbitRadius > 0 ? orbitRadius.toFixed(2) : '自动' }}</span>\n")
        output.append('        </div>\n')
        output.append('        <input\n')
        output.append('          type="range"\n')
        output.append('          :min="0"\n')
        output.append('          :max="20"\n')
        output.append('          :value="orbitRadius"\n')
        output.append('          step="0.1"\n')
        output.append('          @input="orbitRadius = Number($event.target.value)"\n')
        output.append('        />\n')
        output.append('        <!-- 旋转方向切换 -->\n')
        output.append('        <div style="margin-top: 8px;">\n')
        output.append('          <button class="panel-btn panel-btn--ghost orbit-dir-btn" @click="toggleOrbitDirection()">\n')
        output.append("            {{ orbitDirection === 1 ? '逆时针' : '顺时针' }}\n")
        output.append('          </button>\n')
        output.append('        </div>\n')
        output.append('      </template>\n')
        output.append('    </div>\n')
        continue

    # === Insert orbit CSS before quality-hud ===
    if stripped == '.quality-hud {':
        # Insert orbit styles before quality-hud
        output.append('/* ===== Orbit 控制面板样式 ===== */\n')
        output.append('.orbit-panel {\n')
        output.append('  position: absolute;\n')
        output.append('  top: 370px;\n')
        output.append('  right: 18px;\n')
        output.append('  z-index: 60;\n')
        output.append('  width: 210px;\n')
        output.append('  padding: 14px;\n')
        output.append('  display: flex;\n')
        output.append('  flex-direction: column;\n')
        output.append('  gap: 6px;\n')
        output.append('}\n')
        output.append('\n')
        output.append('.orbit-btn-row {\n')
        output.append('  display: flex;\n')
        output.append('  gap: 8px;\n')
        output.append('  margin-top: 8px;\n')
        output.append('}\n')
        output.append('\n')
        output.append('.orbit-dir-btn {\n')
        output.append('  width: 100%;\n')
        output.append('}\n')
        output.append('\n')
        output.append(line)
        i += 1
        continue

    # === Insert orbit mobile responsive styles ===
    if stripped == '  .camera-item {':
        # Check if this is in the mobile media query
        output.append(line)
        i += 1
        # Read until the closing brace of camera-item
        while i < len(lines) and lines[i].rstrip('\n').strip() != '}':
            output.append(lines[i])
            i += 1
        if i < len(lines):
            output.append(lines[i])  # the closing }
            i += 1
        # Add orbit mobile styles
        output.append('\n')
        output.append('  .orbit-panel {\n')
        output.append('    top: 510px;\n')
        output.append('    right: 12px;\n')
        output.append('    width: 180px;\n')
        output.append('    padding: 10px;\n')
        output.append('  }\n')
        continue

    output.append(line)
    i += 1

# Write the result
result = ''.join(output)
with open('3dgs_viewer/spark-3dgs-viewer/src/components/SparkGaussianViewer.vue', 'w', encoding='utf-8') as f:
    f.write(result)

print(f'Written {len(result)} bytes, {result.count(chr(10))} lines')
print('Done')
