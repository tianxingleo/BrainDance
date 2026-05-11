<script setup>
import { ref, onMounted, onBeforeUnmount, watch, computed } from 'vue';
import gsap from 'gsap';

const props = defineProps({
  items: { type: Array, default: () => [] },
  activeId: { type: String, default: '' },
});
const emit = defineEmits(['select']);

const SVG_HEIGHT = 84;
const BASE_Y = 56;     // The flat line Y
const PEAK_Y = 24;     // The Y at the thumb
const BEND_WIDTH = 50; // The horizontal spread of the curve

const containerRef = ref(null);
const width = ref(300);

// Animation State (managed by GSAP)
const state = {
  x: 0,
  bend: 0,     // 0 (flat) to 1 (full curve)
};

const activeIndex = ref(0);

// Pre-load all thumbnails
const loadedThumbs = ref({});

const points = computed(() => {
  const count = Math.max(1, props.items.length);
  return props.items.map((item, i) => {
    const fraction = count > 1 ? i / (count - 1) : 0.5;
    const padding = 40; 
    const w = Math.max(0, width.value - padding * 2);
    return {
      ...item,
      x: padding + fraction * w
    };
  });
});

const pathD = ref('');
const thumbX = ref(0);
const thumbY = ref(BASE_Y);
const formattedTime = ref('');

function updateVisuals() {
  const x = state.x;
  thumbX.value = x;
  
  const ty = BASE_Y + (PEAK_Y - BASE_Y) * state.bend;
  thumbY.value = ty;

  if (state.bend <= 0.01) {
    pathD.value = `M 0 ${BASE_Y} L ${width.value} ${BASE_Y}`;
  } else {
    const lx = Math.max(0, x - BEND_WIDTH);
    const rx = Math.min(width.value, x + BEND_WIDTH);
    const cpL = x - BEND_WIDTH * 0.4;
    const cpR = x + BEND_WIDTH * 0.4;
    
    pathD.value = `M 0 ${BASE_Y} L ${lx} ${BASE_Y} C ${cpL} ${BASE_Y}, ${cpL} ${ty}, ${x} ${ty} C ${cpR} ${ty}, ${cpR} ${BASE_Y}, ${rx} ${BASE_Y} L ${width.value} ${BASE_Y}`;
  }
}

let isDragging = false;
let resizeObserver = null;

onMounted(() => {
  resizeObserver = new ResizeObserver((entries) => {
    width.value = entries[0].contentRect.width;
    snapToActive(false);
  });
  if (containerRef.value) {
    resizeObserver.observe(containerRef.value);
  }
  
  gsap.ticker.add(updateVisuals);
  
  watch(() => props.activeId, () => {
    if (!isDragging) snapToActive(true);
  }, { immediate: true });
});

onBeforeUnmount(() => {
  gsap.ticker.remove(updateVisuals);
  if (resizeObserver && containerRef.value) {
    resizeObserver.unobserve(containerRef.value);
  }
});

function snapToActive(animate = true) {
  const idx = points.value.findIndex(p => p.id === props.activeId);
  if (idx === -1 || points.value.length === 0) return;
  activeIndex.value = idx;
  formattedTime.value = points.value[idx].formattedTime || '';
  
  const targetX = points.value[idx].x;
  
  if (animate) {
    gsap.killTweensOf(state);
    gsap.to(state, {
      x: targetX,
      bend: 1,
      duration: 0.6,
      ease: "elastic.out(1, 0.75)",
    });
  } else {
    state.x = targetX;
    state.bend = 1;
    updateVisuals();
  }
}

function handlePointerDown(e) {
  isDragging = true;
  updateFromEvent(e);
  
  gsap.killTweensOf(state);
  gsap.to(state, {
    bend: 0.65,
    duration: 0.2,
    ease: "power2.out"
  });
  
  window.addEventListener('pointermove', handlePointerMove);
  window.addEventListener('pointerup', handlePointerUp);
  window.addEventListener('touchmove', handleTouchMove, { passive: false });
  window.addEventListener('touchend', handlePointerUp);
}

function handlePointerMove(e) {
  if (!isDragging) return;
  updateFromEvent(e);
}

function handleTouchMove(e) {
  if (!isDragging) return;
  e.preventDefault(); 
  updateFromEvent(e.touches[0]);
}

function updateFromEvent(e) {
  if (!containerRef.value) return;
  const rect = containerRef.value.getBoundingClientRect();
  const tx = Math.max(15, Math.min(width.value - 15, e.clientX - rect.left));
  
  // Find nearest
  let bestIdx = 0;
  let minDist = Infinity;
  points.value.forEach((p, i) => {
    const d = Math.abs(p.x - tx);
    if (d < minDist) {
      minDist = d;
      bestIdx = i;
    }
  });

  const nearest = points.value[bestIdx];
  let finalX = tx;
  if (minDist < 16) {
    finalX = nearest.x + (tx - nearest.x) * 0.2; // stickiness/damping
  }
  
  state.x = finalX;
  activeIndex.value = bestIdx;
  formattedTime.value = points.value[bestIdx].formattedTime || '';
}

function handlePointerUp() {
  if (!isDragging) return;
  isDragging = false;
  window.removeEventListener('pointermove', handlePointerMove);
  window.removeEventListener('pointerup', handlePointerUp);
  window.removeEventListener('touchmove', handleTouchMove);
  window.removeEventListener('touchend', handlePointerUp);
  
  const nearest = points.value[activeIndex.value];
  if (nearest) {
    // 松手时明确抛出选中项，由父组件负责真正触发模型切换。
    emit('select', nearest);
    gsap.killTweensOf(state);
    gsap.to(state, {
      x: nearest.x,
      bend: 1,
      duration: 0.8,
      ease: "elastic.out(1, 0.5)", // bouncy snap
    });
  }
}
</script>

<template>
  <div class="timeline-elastic-wrap" ref="containerRef" @pointerdown="handlePointerDown">
    <!-- Preload Thumbnails -->
    <div style="display: none;">
      <img v-for="item in items" :key="item.id" :src="item.previewImg" loading="eager" @load="loadedThumbs[item.previewImg] = true" />
    </div>

    <svg :width="width" :height="SVG_HEIGHT" class="te-svg" :viewBox="'0 0 ' + width + ' ' + SVG_HEIGHT">
      <path :d="pathD" class="te-path" />
      <circle v-for="(p, i) in points" :key="p.id"
        :cx="p.x" :cy="BASE_Y" r="3"
        class="te-point"
        :class="{ 'te-point-active': i === activeIndex }"
      />
    </svg>
    <div class="te-thumb" :style="{ transform: 'translate3d(' + thumbX + 'px,' + thumbY + 'px,0) translate(-50%,-50%)' }">
      <div class="te-thumb-inner"></div>
      <div class="te-preview-panel" :class="{ 'te-preview-panel-dragging': isDragging }">
        <div v-if="points[activeIndex]?.previewImg" class="te-preview-img-wrap">
          <img :src="points[activeIndex].previewImg" class="te-preview-img" :class="{ 'te-preview-loaded': loadedThumbs[points[activeIndex].previewImg] }" draggable="false" />
        </div>
        <div v-else class="te-preview-empty">未命名</div>
      </div>
      <div class="te-time-text" :class="{ 'te-time-text-dragging': isDragging }">{{ formattedTime }}</div>
    </div>
  </div>
</template>

<style scoped>
.timeline-elastic-wrap {
  position: relative;
  width: 100%;
  height: 112px;
  cursor: grab;
  touch-action: none;
  user-select: none;
  margin: 0 10px 2px;
  flex: 1;
}
.timeline-elastic-wrap:active {
  cursor: grabbing;
}
.te-svg {
  display: block;
  width: 100%;
  height: 100%;
  overflow: visible;
}
.te-path {
  fill: none;
  stroke: var(--card-border, rgba(204, 154, 92, 0.4));
  stroke-width: 2.5;
  stroke-linecap: round;
  transition: stroke 0.3s;
}
.te-point {
  fill: var(--card-border, rgba(107, 122, 143, 0.4));
  transition: all 0.3s;
}
.te-point-active {
  fill: #CC9A5C;
  r: 4;
}
.te-thumb {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  align-items: center;
  z-index: 10;
}
.te-thumb-inner {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #CC9A5C;
  border: 3.5px solid #fff;
  box-shadow: 0 3px 10px rgba(204, 154, 92, 0.5);
  transition: transform 0.2s;
}
.timeline-elastic-wrap:active .te-thumb-inner {
  transform: scale(1.15);
}
.te-preview-panel {
  position: absolute;
  bottom: calc(100% + 12px);
  display: flex;
  flex-direction: column;
  align-items: center;
  background: rgba(30, 30, 32, 0.85);
  padding: 6px 8px;
  border-radius: 10px;
  pointer-events: none;
  transition: opacity 0.2s, transform 0.2s;
  opacity: 0;
  transform: translateY(10px) scale(0.92);
  backdrop-filter: blur(4px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.te-preview-panel::after {
  content: '';
  position: absolute;
  bottom: -4px;
  left: 50%;
  transform: translateX(-50%);
  border-width: 4px 4px 0;
  border-style: solid;
  border-color: rgba(30, 30, 32, 0.85) transparent transparent transparent;
}
.te-preview-panel-dragging, .timeline-elastic-wrap:hover .te-preview-panel {
  opacity: 1;
  transform: translateY(0) scale(1);
}
.te-preview-img-wrap {
  width: 84px;
  height: 60px;
  border-radius: 6px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.1);
}
.te-preview-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0;
  transition: opacity 0.3s;
}
.te-preview-loaded {
  opacity: 1;
}
.te-preview-empty {
  width: 84px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(255, 255, 255, 0.4);
  font-size: 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.1);
}
.te-time-text {
  position: absolute;
  top: calc(100% + 10px);
  left: 50%;
  transform: translateX(-50%);
  color: #fff;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
  text-shadow: 0 1px 3px rgba(0,0,0,0.45);
}
.te-time-text-dragging {
  text-shadow: 0 1px 4px rgba(0,0,0,0.7);
}
</style>
