# BrainDance 电影模式算法逐行精讲

> 源码位置: `3dgs_viewer/my-3dgs-viewer/src/components/GaussianViewer.vue`
> 常量定义在第 44~61 行

```
CINEMATIC_MAX_KEYFRAMES = 18    // 最多 18 个关键帧
CINEMATIC_MIN_KEYFRAMES = 6     // 最少 6 个关键帧
CINEMATIC_MIN_LOOK_AHEAD = 1.2
CINEMATIC_MAX_LOOK_AHEAD = 8.0
CINEMATIC_UP_ALIGNMENT_MIN = 0.45
CINEMATIC_CAMERA_DAMPING_FAST = 0.26
CINEMATIC_CAMERA_DAMPING_SLOW = 0.10
```

---

## 1. 关键帧选取 — selectStableCinematicKeyframes (第 434~495 行)

### 1.0 前置：去重

在进入评分之前，`buildCinematicTrajectory` (第 1207~1215 行) 先做了空间去重：

```javascript
const dedupedKeyframes = [keyframes[0]];
for (let i = 1; i < keyframes.length; i += 1) {
  const prev = dedupedKeyframes[dedupedKeyframes.length - 1];
  const next = keyframes[i];
  // 如果位置距离 < 0.001 且四元数几乎相同，认为是重复的
  const samePoint = prev.position.distanceToSquared(next.position) < 1e-6;
  const sameAngle = Math.abs(prev.quaternion.dot(next.quaternion)) > 0.999999;
  if (samePoint && sameAngle) continue;  // 跳过重复
  dedupedKeyframes.push(next);
}
```

### 1.1 如果关键帧数量 <= 18，直接返回，不做筛选

```javascript
if (keyframes.length <= CINEMATIC_MAX_KEYFRAMES) {
  return keyframes;  // 不超过上限，全部保留
}
```

### 1.2 第一步过滤：淘汰倾斜视角

```javascript
const filtered = keyframes.filter((frame) => {
  const alignment = getCameraUpAlignment(frame.quaternion);
  return alignment >= CINEMATIC_UP_ALIGNMENT_MIN;  // 0.45
});
```

`getCameraUpAlignment` 的含义：相机自身 Y 轴与世界 Y 轴 (0,1,0) 的对齐程度。

```
alignment = |cameraUp · worldUp|

如果相机完全水平: alignment = 1.0 (最稳定)
如果相机倾斜 45°: alignment ≈ 0.707
如果相机完全倒过来: alignment = 1.0 (取了绝对值)
如果相机侧倾严重: alignment 可能接近 0

阈值 0.45 意味着: 容许约 63° 的倾斜,超过就淘汰
```

如果过滤后剩下的帧 < 6 个，则退回到过滤前的全集。

### 1.3 第二步：对每个候选帧计算评分

```javascript
const scored = pool.map((frame, index, arr) => {
  const prev = arr[Math.max(0, index - 1)];       // 前一帧
  const next = arr[Math.min(arr.length - 1, index + 1)]; // 后一帧

  // ---- 指标 A: 水平度 (权重 2.2) ----
  const upAlignment = getCameraUpAlignment(frame.quaternion);
  // 范围 [0, 1], 1 = 完全水平

  // ---- 指标 B: 方向连续性 (权重 1.4) ----
  const prevForward = getCameraForward(prev.quaternion);  // 前一帧的观察方向
  const currForward = getCameraForward(frame.quaternion);  // 当前帧的观察方向
  const nextForward = getCameraForward(next.quaternion);   // 后一帧的观察方向

  const directionalContinuity =
    index > 0 && index < arr.length - 1
      ? Math.max(0, prevForward.dot(currForward)) * 0.5   // 与前一帧方向相似度
      + Math.max(0, currForward.dot(nextForward)) * 0.5    // 与后一帧方向相似度
      : 1;  // 首尾帧给满分

  // dot product 范围 [-1, 1], max(0, ...) 裁掉反向的情况
  // 最终范围 [0, 1]

  // ---- 指标 C: 空间分布 (权重 0.4) ----
  const prevDistance = frame.position.distanceTo(prev.position);
  const nextDistance = frame.position.distanceTo(next.position);
  const avgDistance = (prevDistance + nextDistance) * 0.5;
  // min(avgDistance, 1.5) 防止距离过大的帧获得过高分数

  // ---- 总分 ----
  return {
    frame,
    index,
    score: upAlignment * 2.2       // 水平度最重要
             + directionalContinuity * 1.4  // 方向连续性次之
             + Math.min(avgDistance, 1.5) * 0.4,  // 空间分布最次要
  };
});
```

**评分解读：**

| 指标 | 权重 | 含义 | 为什么要考虑 |
|------|------|------|-------------|
| upAlignment | 2.2 | 相机是否水平 | 倾斜的视角在运镜中会造成旋转不自然 |
| directionalContinuity | 1.4 | 观察方向是否与邻居一致 | 方向突变的帧在运镜中会造成"甩镜" |
| avgDistance | 0.4 | 与邻居帧的平均距离 | 鼓励选取空间上分散的关键帧，避免扎堆 |

### 1.4 第三步：选取得分最高的帧

```javascript
// 强制保留首尾帧（保证路径起点和终点不变）
const forcedIndices = new Set([0, pool.length - 1]);
const selected = scored
  .filter(({ index }) => forcedIndices.has(index))
  .map(({ frame }) => frame);

// 其余帧按得分从高到低排列
const remaining = scored
  .filter(({ index }) => !forcedIndices.has(index))
  .sort((a, b) => b.score - a.score);  // 降序

// 按得分从高到低选取，直到达到 targetCount
for (const candidate of remaining) {
  if (selected.length >= targetCount) break;
  selected.push(candidate.frame);
}

// 恢复原始顺序（按 index 排序）
selected.sort((a, b) => a.index - b.index);
```

### 1.5 兜底机制

如果选出的帧数还是 < 6，则按等间距步进强制补足：

```javascript
if (selected.length < CINEMATIC_MIN_KEYFRAMES) {
  const step = Math.max(1, Math.floor(pool.length / CINEMATIC_MIN_KEYFRAMES));
  for (let i = 0; i < pool.length && selected.length < CINEMATIC_MIN_KEYFRAMES; i += step) {
    const frame = pool[i];
    if (!selected.includes(frame)) selected.push(frame);
  }
  selected.sort((a, b) => a.index - b.index);
}
```

---

## 2. 智能路径规划 — planSmartCinematicRoute (第 578~631 行)

### 2.0 目标

根据关键帧在三维空间中的分布形态，自动判断最适合的运镜风格。

### 2.1 特征提取

```javascript
// ---- 特征 1: 水平距离统计 ----
// 每个关键帧到场景中心的水平距离（忽略 Y 轴）
const horizontalDistances = keyframes.map((frame) => {
  const offset = frame.position.clone().sub(worldCenter);
  offset.y = 0;               // 去掉高度
  return offset.length();      // 只看水平距离
});

const radiiMean = horizontalDistances.reduce(...) / count;    // 平均半径
const radiiVariance = ...reduce(sum, (r - mean)²) / count;   // 方差
const radiiStd = Math.sqrt(radiiVariance);                    // 标准差

// radiiStd 小 → 所有关键帧离中心差不多远 → 它们近似在一个圆柱面上
// radiiStd 大 → 关键帧离中心的距离差异大 → 不适合环绕
```

```javascript
// ---- 特征 2: 高度分布 ----
const yValues = keyframes.map(frame => frame.position.y);
const heightSpread = Math.max(...yValues) - Math.min(...yValues);
// heightSpread 大 → 关键帧在高度上分布范围大 → 适合从下往上或从上往下的摇臂运镜
```

```javascript
// ---- 特征 3: 角度分布 ----
// 每个关键帧相对于场景中心的水平角度
const angles = keyframes.map((frame) => {
  const offset = frame.position.clone().sub(worldCenter);
  return Math.atan2(offset.z, offset.x);  // [-π, π]
});

// 核心难点：角度是环状的，-179° 和 +179° 实际只差 2°
// unwrapCircularAngles 解决这个问题
const unwrappedAngles = unwrapCircularAngles(angles);
const angleSpread = unwrappedAngles[unwrappedAngles.length - 1].angle
                  - unwrappedAngles[0].angle;
// angleSpread 大 → 关键帧在水平面上绕了一圈 → 适合环绕
```

### 2.2 unwrapCircularAngles 详解 (第 497~526 行)

这是处理角度环绕断裂的经典算法：

```
假设有 5 个角度: [-170°, -30°, 10°, 50°, 160°]

Step 1: 排序 → [-170°, -30°, 10°, 50°, 160°]

Step 2: 找最大间隙
  -170 → -30   间隙 140°
  -30  → 10    间隙 40°
  10   → 50    间隙 40°
  50   → 160   间隙 110°
  160  → -170  间隙 330° (跨过 ±180° 边界)  ← 最大间隙!

Step 3: 从最大间隙处"切开"并展开
  以 -170° 为起点重排:
  -170° → -30° → 10° → 50° → 160° → 190°(=-170+360)

Step 4: angleSpread = 190° - (-170°) = 360°
  → 这是一个完整环绕！
```

### 2.3 三种模式的判定规则

```javascript
let routeMode = 'dolly';  // 默认推拉模式

if (angleSpread > 1.1 && radiiStd < Math.max(0.35, radiiMean * 0.28)) {
  // 条件 1: 角度分布 > 63° (1.1 弧度)
  //   → 关键帧在水平面上分散较广
  // 条件 2: 半径标准差 < 平均半径 * 28%
  //   → 所有关键帧离中心差不多远
  // 两个条件同时满足 → 它们在近似同一半径的圆上 → 环绕模式
  routeMode = 'orbit';

} else if (heightSpread > Math.max(0.8, radiiMean * 0.42)) {
  // 条件: 高度差 > 平均水平半径 * 42%
  //   → 关键帧在垂直方向上的分布比水平方向更显著
  // → 摇臂模式（从低处到高处的升降运镜）
  routeMode = 'crane';

} else {
  // 其他情况 → 推拉模式（沿主轴方向移动）
  routeMode = 'dolly';
}
```

**图示理解：**

```
Orbit (环绕)             Crane (摇臂)            Dolly (推拉)
  场景中心                   场景中心               场景中心
    ●                          ●                      ●
   /|\                        6                       1
  / | \                      5                        2
 1  2  3                    4                         3
  \ | /                    3                          4
   \|/                     2                           5
    圆                     1                            6
                            |                        沿一条线
                           升降
```

### 2.4 各模式的排序策略

```javascript
if (routeMode === 'orbit') {
  // 按角度排序（绕中心一圈）
  const byAngle = unwrappedAngles.map(({ index }) => keyframes[index]);
  ordered = chooseLowerCostRouteDirection(byAngle, worldCenter);

} else if (routeMode === 'crane') {
  // 按高度排序（从低到高或从高到低）
  ordered = chooseLowerCostRouteDirection(
    keyframes.slice().sort((a, b) => a.position.y - b.position.y),
    worldCenter
  );

} else {
  // dolly: 找出"最长的水平方向"作为主轴，然后沿主轴投影排序
  const axis = getDominantHorizontalAxis(keyframes);
  ordered = chooseLowerCostRouteDirection(
    keyframes.slice().sort((a, b) =>
      a.position.dot(axis) - b.position.dot(axis)
    ),
    worldCenter
  );
}
```

`getDominantHorizontalAxis` 找所有关键帧对中距离最远的那一对，连线方向就是主轴。

---

## 3. 方向优化 — chooseLowerCostRouteDirection (第 562~576 行)

### 3.1 问题

排序后的关键帧可以正向走 A→B→C→D，也可以反向走 D→C→B→A。两种方向的视觉体验完全不同。

### 3.2 转移代价函数 — computeRouteTransitionCost (第 528~538 行)

```javascript
const computeRouteTransitionCost = (from, to, worldCenter) => {
  // 因素 1: 空间距离 (权重 1.25)
  const distance = from.position.distanceTo(to.position);

  // 因素 2: 观察方向突变 (权重 1.4)
  const fromForward = getCameraForward(from.quaternion);  // from 的观察方向
  const toForward = getCameraForward(to.quaternion);      // to 的观察方向
  const forwardMismatch = 1 - Math.max(-1, Math.min(1, fromForward.dot(toForward)));
  // dot=1 (同向) → mismatch=0 (好)
  // dot=0 (垂直) → mismatch=1 (差)
  // dot=-1 (反向) → mismatch=2 (最差)

  // 因素 3: 焦点方向突变 (权重 0.9)
  const fromTargetDir = worldCenter.clone().sub(from.position).normalize();
  const toTargetDir = worldCenter.clone().sub(to.position).normalize();
  const focusMismatch = 1 - Math.max(-1, Math.min(1, fromTargetDir.dot(toTargetDir)));
  // 两帧看向中心的方向是否一致

  // 因素 4: 高度跳变 (权重 0.35)
  const heightDelta = Math.abs(from.position.y - to.position.y);

  return distance * 1.25
       + forwardMismatch * 1.4
       + focusMismatch * 0.9
       + heightDelta * 0.35;
};
```

**为什么要考虑这么多因素？**

假设只看距离：
- A→B 距离很近，但 A 朝北 B 朝南 → 运镜会突然 180° 掉头
- A→C 距离远一点，但 A 和 C 朝向一致 → 运镜更流畅

所以方向连续性的权重 (1.4) 比距离 (1.25) 更高。

### 3.3 双向比较

```javascript
const chooseLowerCostRouteDirection = (orderedKeyframes, worldCenter) => {
  const routeCost = (frames) => {
    let total = 0;
    for (let i = 1; i < frames.length; i += 1) {
      total += computeRouteTransitionCost(frames[i - 1], frames[i], worldCenter);
    }
    return total;  // 累加所有相邻帧的转移代价
  };

  const forward  = orderedKeyframes.slice();          // 正向
  const reverse  = orderedKeyframes.slice().reverse(); // 反向

  return routeCost(forward) <= routeCost(reverse) ? forward : reverse;
  // 取总代价更小的方向
};
```

**举例：**

```
关键帧: A(高) → B(中) → C(低)

正向 A→B→C:
  cost(A,B) = distance*1.25 + (方向变化)*1.4 + ... 
  cost(B,C) = distance*1.25 + (方向变化)*1.4 + ...
  total_forward = cost(A,B) + cost(B,C)

反向 C→B→A:
  cost(C,B) = ... (可能与 cost(B,C) 不同，因为方向翻转了)
  cost(B,A) = ...
  total_reverse = cost(C,B) + cost(B,A)

如果 total_forward < total_reverse → 选正向
否则 → 选反向
```

---

## 4. 多级平滑 (第 318~422 行)

平滑是分多个阶段串联进行的，每个阶段针对不同类型的数据。

### 4.1 第一级：四元数连续性保证 — ensureQuaternionContinuity (第 380~396 行)

**问题**：四元数 q 和 -q 表示相同的旋转。如果相邻帧的四元数符号不一致，slerp 会走"远路"（绕大圈而不是小圈）。

```javascript
const ensureQuaternionContinuity = (quaternions) => {
  const result = [quaternions[0].clone().normalize()];
  for (let i = 1; i < quaternions.length; i += 1) {
    const next = quaternions[i].clone().normalize();
    // 如果与前一个四元数的点积 < 0，说明它们在超球面的对面
    if (result[i - 1].dot(next) < 0) {
      // 翻转符号（等价旋转，但连续性更好）
      next.x *= -1;
      next.y *= -1;
      next.z *= -1;
      next.w *= -1;
    }
    result.push(next);
  }
  return result;
};
```

**图示：**
```
修正前:                        修正后:
q0(+) ---- q1(-)               q0(+) ---- q1(+)
  |            |                  |           |
q2(+) ---- q3(-)               q2(+) ---- q3(+)

slerp(q0, q1):                  slerp(q0, q1):
dot < 0 → 走远路 180°           dot > 0 → 走近路 ~10°
```

### 4.2 第二级：四元数 slerp 低通滤波 — smoothQuaternionSeries (第 398~422 行)

```javascript
const smoothQuaternionSeries = (quaternions, amount) => {
  // amount = cinematicSmoothness (用户可调，范围 [0, 1])

  const strength = clamp(amount, 0, 1);
  const passes = Math.max(1, Math.round(1 + strength * 3));  // 1~4 次迭代
  const blend = 0.16 + strength * 0.22;                       // 0.16~0.38

  let result = ensureQuaternionContinuity(quaternions);

  for (let pass = 0; pass < passes; pass += 1) {
    const nextSeries = result.map((quat, index) => {
      if (index === 0 || index === result.length - 1) return quat.clone();
      // 首尾帧不动（锚点）

      const prev = result[index - 1];
      const curr = result[index];
      const next = result[index + 1];

      // 先计算 prev 和 next 的中间值
      const averaged = prev.clone().slerp(next, 0.5);
      // 然后把 curr 向这个中间值拉一点
      return curr.slerp(averaged, blend).normalize();
    });
    result = ensureQuaternionContinuity(nextSeries);
    // 每次迭代后重新确保连续性
  }

  return result;
};
```

**直觉理解**：每次迭代把中间帧的四元数朝"左右邻居的平均值"拉一点。多次迭代后，突变的朝向会被逐渐磨平。

```
原始:  N → N → S → S → S     (第 3 帧突然从北转向南)
1 pass: N → N → NNE → SSE → S  (第 3 帧被拉向中间)
2 pass: N → NNE → NE → ESE → S (进一步平滑)
3 pass: N → NNE → E → ESE → S  (接近均匀过渡)
```

### 4.3 第三级：位置向量平滑 — smoothVectorSeries (第 318~339 行)

```javascript
const smoothVectorSeries = (vectors, amount) => {
  const strength = clamp(amount, 0, 1);
  const passes = Math.max(1, Math.round(1 + strength * 3));  // 1~4 次
  const blend = 0.12 + strength * 0.26;                       // 0.12~0.38

  let result = vectors.map(vec => vec.clone());

  for (let pass = 0; pass < passes; pass += 1) {
    const nextSeries = result.map((vec, index) => {
      if (index === 0 || index === result.length - 1) return vec.clone();

      // 三点加权均值: (左 + 中*2 + 右) / 4
      const blended = result[index - 1].clone()
        .add(result[index].clone().multiplyScalar(2))
        .add(result[index + 1])
        .multiplyScalar(0.25);

      // 与原始值 lerp 混合
      return vec.clone().lerp(blended, blend);
    });
    result = nextSeries;
  }

  return result;
};
```

**直觉**：等价于对位置序列做一个低通滤波器。`blend` 越大，平滑力度越大。

### 4.4 第四级：焦距标量平滑 — smoothScalarSeries (第 341~359 行)

与位置平滑完全相同的算法，只是作用于 1D 标量值（焦距 `fl_y`）：

```javascript
const smoothScalarSeries = (values, amount) => {
  // ... 同样的 passes/blend 计算
  for (let pass = 0; pass < passes; pass += 1) {
    const nextSeries = result.map((value, index) => {
      if (index === 0 || index === result.length - 1) return value;
      const averaged = (result[index - 1] + result[index] * 2 + result[index + 1]) / 4;
      return THREE.MathUtils.lerp(value, averaged, blend);
    });
    result = nextSeries;
  }
  return result;
};
```

### 4.5 makeUprightQuaternion — 朝向修正 (第 369~378 行)

在平滑之前，每个关键帧的四元数都被"修正为水平朝向"：

```javascript
const makeUprightQuaternion = (position, quaternion, fallbackTarget) => {
  const forward = getCameraForward(quaternion);
  let target = position.clone().add(forward);  // 看向的点

  // 用 lookAt 重新构造一个"保持观察方向但强制水平"的四元数
  return makeLookQuaternion(position, target);
};
```

这一步确保运镜过程中相机不会出现明显的侧倾。

---

## 5. CatmullRom 曲线插值与等弧长采样 — buildCinematicSegment + sampleCinematicTrajectory

### 5.1 为什么用 CatmullRom？

CatmullRom 曲线是**过控制点的样条曲线**（不像贝塞尔曲线只逼近）。三个参数化方式：

| 参数化 | 特点 |
|--------|------|
| `uniform` | 等参数间距，但曲线速度不均匀（弯道处会加速） |
| `chordal` | 按弦长参数化，但尖角处可能产生尖刺 |
| **`centripetal`** | 按向心参数化，**既不会产生尖刺也不会产生交叉** |

本项目选用 `centripetal`，这是最稳定的选择。

### 5.2 构建曲线 — buildCinematicSegment (第 633~686 行)

```javascript
// 位置曲线
const curve = new THREE.CatmullRomCurve3(
  preparedKeyframes.map(frame => frame.position.clone()),
  false,          // 不闭合
  'centripetal'   // 向心参数化
);

// 注视目标曲线（另一条独立的 CatmullRom）
const lookCurve = new THREE.CatmullRomCurve3(
  preparedKeyframes.map(frame => frame.target.clone()),
  false,
  'centripetal'
);

// 计算累积距离（用于等弧长参数化）
const cumulativeDistances = [0];
for (let i = 1; i < preparedKeyframes.length; i += 1) {
  cumulativeDistances.push(
    cumulativeDistances[i - 1] + prev.position.distanceTo(next.position)
  );
}
```

### 5.3 注视目标的计算 (buildCinematicTrajectory 第 1229~1245 行)

注视目标不是简单的场景中心，而是相机前方向与场景中心的混合：

```javascript
const lookTargets = orderedKeyframes.map((frame, index) => {
  // 1. 相机前方向延伸到远处
  const forward = new THREE.Vector3(0, 0, -1)
    .applyQuaternion(frame.quaternion)
    .normalize();
  const frameDistanceToCenter = rawPositions[index].distanceTo(worldCenter);
  const forwardTarget = rawPositions[index].clone().add(
    forward.multiplyScalar(Math.max(2.2, frameDistanceToCenter * 0.9))
  );

  // 2. 如果开启了主体锁定，混合场景中心
  if (!cinematicSubjectLock.value) return forwardTarget;

  return forwardTarget.lerp(
    worldCenter,
    clamp(0.48 + cinematicSmoothness.value * 0.26, 0, 0.9)
    // blend 系数: 0.48~0.74
    // smoothness 越大 → 越偏向场景中心 → 运镜越稳定
  );
});
```

### 5.4 等弧长采样 — sampleCinematicTrajectory (第 1272~1312 行)

关键问题：`curve.getPoint(t)` 的 t 是参数空间而不是弧长空间。如果直接用 t = elapsed/duration，相机在曲线弯曲处会"加速"（视觉上速度不均匀）。

本项目通过 `curve.getPointAt(t)` 实现等弧长参数化（Three.js 内部会做弧长重参数化），同时四元数和注视目标通过累积距离做分段插值：

```javascript
const sampleCinematicTrajectory = (trajectory, normalizedT) => {
  const t = clamp(normalizedT, 0, 1);

  // 1. 等弧长取位置
  const position = trajectory.curve.getPointAt(t);
  //       ↑ getPointAt 是弧长参数化，getPoint 是原始参数化

  // 2. 通过累积距离确定落在哪两个关键帧之间
  const distanceAlongPath = trajectory.totalDistance * t;
  let segmentIndex = ...; // 二分查找或线性扫描

  // 3. 计算段内局部参数
  const localT = smootherstep(
    (distanceAlongPath - startDistance) / segmentLength,
    0, 1
  );
  // smootherstep = 6t⁵ - 15t⁴ + 10t³
  // 是 smoothstep 的更高阶版本，起点和终点的速度和加速度都为 0

  // 4. 四元数: 前后关键帧 slerp
  const quaternion = from.stabilizedQuaternion
    .clone()
    .slerp(to.stabilizedQuaternion, localT)
    .normalize();

  // 5. 注视目标: 前后目标 lerp
  const target = from.target.clone().lerp(to.target, localT);

  // 6. 焦距: 前后焦距 lerp
  const fl_y = lerp(from.fl_y, to.fl_y, localT);

  return { position, quaternion, target, fl_y, nearestPoseIndex };
};
```

**smootherstep 的作用**：

```
smoothstep(t) = 3t² - 2t³          (一阶导数在端点为 0)
smootherstep(t) = 6t⁵ - 15t⁴ + 10t³ (一阶和二阶导数在端点都为 0)

效果: 在关键帧之间的过渡更加平滑，不会有加加速度(jerk)
```

---

## 6. 阻尼跟踪 — applyCinematicSample (第 1314~1364 行)

### 6.1 为什么需要阻尼？

即使曲线采样已经很平滑，如果直接把采样值赋给相机，帧率波动会导致视觉上的抖动。阻尼跟踪让相机"柔和地追踪"采样点。

### 6.2 阻尼系数计算

```javascript
const dampingAlpha = THREE.MathUtils.lerp(
  CINEMATIC_CAMERA_DAMPING_FAST,  // 0.26 — 跟踪快，接近直接赋值
  CINEMATIC_CAMERA_DAMPING_SLOW,  // 0.10 — 跟踪慢，更柔和
  cinematicSmoothness.value       // 用户可调参数 [0, 1]
);
// smoothness=0 → alpha=0.26 (快速响应)
// smoothness=1 → alpha=0.10 (缓慢追踪)
```

### 6.3 一阶低通滤波

```javascript
// 位置: lerp (线性插值)
filteredPosition.lerp(samplePosition, dampingAlpha);
// 等价于: filtered = filtered * (1-alpha) + sample * alpha

// 旋转: slerp (球面线性插值)
filteredQuaternion.slerp(sampleQuaternion, dampingAlpha).normalize();
// 四元数不能直接 lerp，必须用 slerp 在超球面上插值

// 焦距: lerp，额外乘 0.85 使焦距变化更缓慢
filteredFocal = lerp(current, target, dampingAlpha * 0.85);
```

### 6.4 每帧执行

```javascript
// 赋值给实际相机
cam.position.copy(filteredPosition);
cam.quaternion.copy(filteredQuaternion);

// 如果有焦距信息，更新相机 FOV
if (filteredSample.fl_y && filteredSample.h) {
  applyFocalLengthPx(filteredSample.fl_y);
}
```

**直觉**：想象采样点是一个跑步的人，相机是用弹性绳牵着的一只狗。阻尼系数就是弹性绳的弹性——弹性大 (alpha 大)，狗跟得紧但可能抖；弹性小 (alpha 小)，狗跟得慢但很平稳。

---

## 7. 循环桥接 — buildLoopBridgeSegment (第 688~743 行)

### 7.1 问题

运镜从第 1 帧走到最后一帧后，如果循环播放，需要从最后一帧回到第 1 帧。直接跳回去会很不自然，需要构建一条过渡路径。

### 7.2 桥接路径构造

```javascript
const first = mainSegment.keyframes[0];  // 起点
const last = mainSegment.keyframes[last]; // 终点

// ---- 计算"抬高量"和"外推量" ----
const directDistance = last.position.distanceTo(first.position);

const liftAmount = Math.max(
  sceneRadius * 0.55,           // 至少半个场景半径高
  directDistance * 0.22,         // 或首尾距离的 22%
  0.9                            // 最低 0.9 单位
);
const radialPush = Math.max(
  sceneRadius * 0.18,           // 向外推一点
  directDistance * 0.08,
  0.35
);
```

### 7.3 四个桥接控制点

```javascript
// 从终点方向外推（水平分量）
const startOut = (last.position - worldCenter).setY(0).normalize() * radialPush;
// 从起点方向外推（水平分量）
const endOut = (first.position - worldCenter).setY(0).normalize() * radialPush;

// 场景中心略抬高
const centerLift = worldCenter + (0, sceneRadius * 0.15, 0);

const bridgePositions = [
  last.position,                                          // P0: 从终点出发
  last.position + (0, liftAmount, 0) + startOut,          // P1: 升高 + 向外推
  first.position + (0, liftAmount * 0.86, 0) + endOut,   // P2: 接近起点 + 略低
  first.position,                                         // P3: 降落回起点
];
```

**三维空间中的路径形状：**

```
                    P1 (终点上空，外推)
                   /  \
                  /    \
                 /      \        ← 空中过渡弧
                /        \
    P0 (终点)              P2 (起点上空，外推)
                              \
                               \
                                P3 (起点)
```

### 7.4 注视目标过渡

```javascript
const bridgeTargets = [
  last.target.clone().lerp(centerLift, 0.4),  // 从终点目标转向中心
  centerLift.clone(),                          // 注视中心
  centerLift.clone(),                          // 注视中心
  first.target.clone().lerp(centerLift, 0.28), // 从中心转向起点目标
];
```

**效果**：过渡时相机先抬头看向场景中心上方，然后逐渐低头回到起点的注视方向。

### 7.5 桥接时长

```javascript
const bridgeDurationMs = clamp(
  directDistance * 1350 + 1800,  // 基础时长
  2400,   // 最短 2.4 秒
  6200    // 最长 6.2 秒
) / cinematicSpeed;
```

### 7.6 播放状态机 (stepCinematicPlayback 第 1366~1414 行)

```javascript
// 两段式播放:
// phase 'main'        → 主运镜路径（关键帧 A→B→...→N）
// phase 'loop-bridge' → 桥接路径（N → 空中 → A）

if (normalizedT >= 1) {
  if (phase === 'loop-bridge') {
    // 桥接结束 → 切回主路径
    phase = 'main';
    startTimeMs = now;
  } else if (loop && loopBridge) {
    // 主路径结束 → 切到桥接
    phase = 'loop-bridge';
    startTimeMs = now;
  } else if (loop) {
    // 没有桥接但需要循环 → 重新播放主路径
    phase = 'main';
    startTimeMs = now;
  } else {
    // 不循环 → 停在最后一帧
    normalizedT = 1;
    stopCinematicPlayback();
  }
}
```

**完整循环的时间线：**

```
|←───── main phase ──────→|←── loop-bridge ──→|←───── main phase ──────→|
A → B → C → ... → N       N ↗ 空中 ↘ A       A → B → C → ... → N
7s ~ 42s                   2.4s ~ 6.2s         7s ~ 42s
```

---

## 总结：完整管线流程图

```
原始位姿列表 (cameraPoses)
        │
        ▼
  ┌─ getPreferredCinematicPoses ─┐
  │ 选出有标签的前 12 个位姿      │
  └──────────────┬──────────────┘
                 │
        ▼
  ┌─ resolvePoseCameraState ─┐
  │ 从 matrix 解析出          │
  │ position + quaternion     │
  └──────────────┬───────────┘
                 │
        ▼
  ┌─ makeUprightQuaternion ──┐
  │ 修正为水平朝向            │
  └──────────────┬───────────┘
                 │
        ▼
  ┌─ 去重 ──────────────────┐
  │ 去掉位置和朝向都相同的帧  │
  └──────────────┬───────────┘
                 │
  ┌──────────────▼──────────────┐
  │  STEP 1: 评分选取关键帧     │
  │  score = 水平度*2.2         │
  │        + 方向连续*1.4       │
  │        + 空间分布*0.4       │
  │  取 top 6~18 个             │
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  STEP 2: 智能路径规划       │
  │  分析 angleSpread           │
  │       radiiStd              │
  │       heightSpread          │
  │  → orbit / crane / dolly    │
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  STEP 3: 方向优化           │
  │  正向 vs 反向               │
  │  取转移代价更小的方向       │
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  STEP 4: 多级平滑           │
  │  4a. 四元数连续性保证       │
  │  4b. slerp 低通滤波 (1~4遍)│
  │  4c. 位置加权均值平滑       │
  │  4d. 焦距标量平滑           │
  │  4e. 注视目标平滑           │
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  STEP 5: CatmullRom 曲线    │
  │  centripetal 参数化          │
  │  位置曲线 + 注视目标曲线     │
  │  getPointAt(t) 等弧长采样   │
  │  smootherstep 段内缓动      │
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  STEP 6: 阻尼跟踪           │
  │  position: lerp(α=0.10~0.26)│
  │  quaternion: slerp(α)       │
  │  focal: lerp(α*0.85)        │
  └──────────────┬──────────────┘
                 │
  ┌──────────────▼──────────────┐
  │  STEP 7: 循环桥接           │
  │  终点 → 升高外推            │
  │       → 空中过渡(看中心)    │
  │       → 降落回起点          │
  │  时长 2.4~6.2 秒            │
  └─────────────────────────────┘
```
