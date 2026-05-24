<script setup>
import { computed, ref, watch, nextTick, onMounted, onBeforeUnmount, shallowRef } from 'vue';

const props = defineProps({
  models: { type: Array, default: () => [] },
  activeModelId: { type: String, default: '' },
  poses: { type: Array, default: () => [] },
  activePoseId: { type: String, default: '' },
  searchQuery: { type: String, default: '' },
  getPosePresentationId: { type: Function, default: (p) => p.id },
  hasModels: { type: Boolean, default: false },
  hasPoses: { type: Boolean, default: false },
});

const emit = defineEmits(['selectModel', 'selectPose']);

const mode = ref('pose'); // 'pose' | 'model'
const scrollRef = ref(null);
const loadedThumbs = ref({});

let isDragging = false;
let dragStartX = 0;
let scrollStartLeft = 0;
let dragMoved = false;

const modelItems = shallowRef([]);

const showTabs = computed(() => props.models.length > 0);

watch([() => props.hasModels, () => props.hasPoses], () => {
  if (mode.value === 'model' && !props.hasModels) mode.value = 'pose';
}, { immediate: true });

watch(mode, () => {
  dragMoved = false;
});

function formatTime(dateStr) {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${mm}/${dd} ${hh}:${min}`;
}

watch(
  () => props.models,
  (models) => {
    modelItems.value = [...models]
      .sort((a, b) => {
        const ta = new Date(a.createdAt || 0).getTime();
        const tb = new Date(b.createdAt || 0).getTime();
        return tb - ta;
      })
      .map((model) => ({
        ...model,
        formattedTime: formatTime(model.createdAt),
        thumb: model.previewImg || model.previewImage || model.preview_url || model.preview || '',
      }));
  },
  { immediate: true },
);

function onClickModel(model) {
  if (dragMoved) return;
  if (model.id === props.activeModelId) return;
  emit('selectModel', model);
}

function onClickPose(pose) {
  if (dragMoved) return;
  emit('selectPose', pose);
}

function scrollToActive() {
  if (!scrollRef.value) return;
  const el = scrollRef.value.querySelector('.bs-item--active');
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
  }
}

watch([() => props.activePoseId, () => props.activeModelId, mode], () => {
  nextTick(scrollToActive);
});

function onPointerDown(e) {
  isDragging = true;
  dragMoved = false;
  dragStartX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
  scrollStartLeft = scrollRef.value ? scrollRef.value.scrollLeft : 0;
}

function onPointerMove(e) {
  if (!isDragging || !scrollRef.value) return;
  const x = e.clientX || (e.touches && e.touches[0].clientX) || 0;
  const dx = x - dragStartX;
  if (Math.abs(dx) > 3) dragMoved = true;
  scrollRef.value.scrollLeft = scrollStartLeft - dx;
}

function onPointerUp() {
  isDragging = false;
  // 留半帧给 click 判断 dragMoved，再清掉
  setTimeout(() => { dragMoved = false; }, 0);
}

onMounted(() => {
  window.addEventListener('mouseup', onPointerUp);
  window.addEventListener('touchend', onPointerUp);
});
onBeforeUnmount(() => {
  window.removeEventListener('mouseup', onPointerUp);
  window.removeEventListener('touchend', onPointerUp);
});
</script>

<template>
  <div class="bs-root"
    @mousedown.stop @touchstart.stop @touchmove.stop @touchend.stop @wheel.stop>
    <div class="bs-tabs" v-if="showTabs">
      <button
        class="bs-tab" :class="{ 'bs-tab--active': mode === 'pose' }"
        @click="mode = 'pose'"
      >空间</button>
      <button
        class="bs-tab" :class="{ 'bs-tab--active': mode === 'model' }"
        @click="mode = 'model'"
      >时间</button>
    </div>

    <div class="bs-track-wrap">
      <div class="bs-track"
        ref="scrollRef"
        @mousedown="onPointerDown"
        @mousemove="onPointerMove"
        @touchstart="onPointerDown"
        @touchmove="onPointerMove"
      >
        <template v-if="mode === 'pose'">
          <div
            v-for="pose in poses"
            :key="pose.id"
            class="bs-item"
            :class="{ 'bs-item--active': activePoseId === getPosePresentationId(pose) }"
            @click="onClickPose(pose)"
          >
            <img
              v-if="pose.image_url"
              :src="pose.image_url"
              class="bs-thumb"
              :class="{ 'bs-thumb--loaded': loadedThumbs[pose.image_url] }"
              @load="loadedThumbs[pose.image_url] = true"
              draggable="false"
              loading="eager"
              decoding="async"
              fetchpriority="low"
            />
            <div v-else class="bs-thumb bs-thumb--empty">
              <span>未命名</span>
            </div>
          </div>
        </template>

        <template v-if="mode === 'model'">
          <div
            v-for="model in modelItems"
            :key="model.id"
            class="bs-item bs-item--model"
            :class="{ 'bs-item--active': activeModelId === model.id }"
            @click="onClickModel(model)"
          >
            <img
              v-if="model.thumb"
              :src="model.thumb"
              class="bs-thumb"
              :class="{ 'bs-thumb--loaded': loadedThumbs[model.thumb] }"
              @load="loadedThumbs[model.thumb] = true"
              draggable="false"
              loading="eager"
              decoding="async"
            />
            <div v-else class="bs-thumb bs-thumb--empty">
              <span>未命名</span>
            </div>
            <div class="bs-time">{{ model.formattedTime }}</div>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bs-root {
  position: absolute;
  bottom: 24px;
  left: 16px;
  right: 16px;
  z-index: 100;
  display: flex;
  align-items: center;
  gap: 12px;
  pointer-events: auto;
  background: linear-gradient(180deg, rgba(249, 249, 248, 0.72) 0%, rgba(249, 249, 248, 0.88) 100%);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-radius: 18px;
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  box-shadow: 0 8px 22px var(--card-shadow, rgba(0, 0, 0, 0.1));
  padding: 12px 14px;
}

.bs-tabs {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex-shrink: 0;
  align-self: center;
}

.bs-tab {
  padding: 6px 12px;
  border: none;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.1));
  color: var(--text-muted, rgba(30, 30, 32, 0.55));
  white-space: nowrap;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  outline: none;
}
.bs-tab:focus-visible {
  outline: none;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.55);
}
.bs-tab--active {
  background: rgba(204, 154, 92, 0.9);
  color: #fff;
  box-shadow: 0 2px 8px rgba(204, 154, 92, 0.3);
}
.bs-tab:hover:not(.bs-tab--active) {
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.18));
  color: var(--text-secondary, rgba(30, 30, 32, 0.78));
}

.bs-track-wrap {
  position: relative;
  flex: 1;
  min-width: 0;
  overflow: hidden;
}

.bs-track {
  display: flex;
  gap: 10px;
  align-items: center;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 6px 28px;
  scrollbar-width: none;
  -ms-overflow-style: none;
  cursor: grab;
  user-select: none;
  -webkit-mask-image: linear-gradient(to right, transparent, black 24px, black calc(100% - 24px), transparent);
  mask-image: linear-gradient(to right, transparent, black 24px, black calc(100% - 24px), transparent);
  -webkit-tap-highlight-color: transparent;
}
.bs-track::-webkit-scrollbar {
  display: none;
}
.bs-track:active {
  cursor: grabbing;
}

.bs-item {
  position: relative;
  width: 84px;
  height: 56px;
  flex-shrink: 0;
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
  transition:
    transform 0.22s cubic-bezier(0.22, 1, 0.36, 1),
    opacity 0.22s ease,
    border-color 0.22s ease,
    box-shadow 0.22s ease;
  border: 2px solid var(--card-border, rgba(107, 122, 143, 0.12));
  opacity: 0.74;
  box-shadow: 0 2px 8px var(--card-shadow, rgba(0, 0, 0, 0.08));
  transform: scale(0.92);
  transform-origin: center;
  outline: none;
  -webkit-tap-highlight-color: transparent;
}
.bs-item:focus-visible {
  outline: none;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.55), 0 2px 8px var(--card-shadow, rgba(0, 0, 0, 0.08));
}
.bs-item--active {
  border-color: #CC9A5C;
  opacity: 1;
  box-shadow: 0 4px 14px rgba(204, 154, 92, 0.32);
  transform: scale(1.08);
}
.bs-item--active:focus-visible {
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.55), 0 4px 14px rgba(204, 154, 92, 0.32);
}
.bs-item:hover:not(.bs-item--active) {
  opacity: 0.92;
  border-color: var(--input-focus-border, rgba(107, 122, 143, 0.28));
}

.bs-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
  -webkit-tap-highlight-color: transparent;
  background: linear-gradient(135deg, rgba(60, 60, 66, 0.35) 0%, rgba(40, 40, 46, 0.45) 100%);
  opacity: 0;
  transition: opacity 0.4s ease;
}
.bs-thumb.bs-thumb--loaded {
  opacity: 1;
}
.bs-thumb--empty {
  opacity: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(255, 255, 255, 0.55);
  font-size: 10px;
}

/* 时间 tab：active 卡底部叠时间戳 */
.bs-item--model .bs-time {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 2px 6px;
  font-size: 10px;
  font-weight: 600;
  color: #fff;
  text-align: center;
  line-height: 1.3;
  background: linear-gradient(transparent, rgba(0, 0, 0, 0.7));
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s ease;
}
.bs-item--model.bs-item--active .bs-time,
.bs-item--model:hover .bs-time {
  opacity: 1;
}

[data-theme="dark"] .bs-root {
  background: linear-gradient(
    180deg,
    rgba(22, 24, 30, 0.78) 0%,
    rgba(22, 24, 30, 0.92) 100%
  );
  border-color: rgba(255, 255, 255, 0.08);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
}

[data-theme="dark"] .bs-tab {
  background: rgba(255, 255, 255, 0.06);
  color: rgba(245, 247, 250, 0.6);
}
[data-theme="dark"] .bs-tab:hover:not(.bs-tab--active) {
  background: rgba(255, 255, 255, 0.12);
  color: rgba(245, 247, 250, 0.92);
}
[data-theme="dark"] .bs-tab--active {
  background: rgba(204, 154, 92, 0.92);
  color: #1a1a1f;
  box-shadow: 0 2px 8px rgba(204, 154, 92, 0.35);
}
[data-theme="dark"] .bs-tab:focus-visible {
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.18);
}

[data-theme="dark"] .bs-item {
  border-color: rgba(255, 255, 255, 0.08);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
}
[data-theme="dark"] .bs-item:focus-visible {
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.2),
    0 2px 8px rgba(0, 0, 0, 0.5);
}
[data-theme="dark"] .bs-item:hover:not(.bs-item--active) {
  border-color: rgba(255, 255, 255, 0.22);
}
[data-theme="dark"] .bs-item--active {
  border-color: #CC9A5C;
  box-shadow: 0 4px 14px rgba(204, 154, 92, 0.4);
}
[data-theme="dark"] .bs-item--active:focus-visible {
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.22),
    0 4px 14px rgba(204, 154, 92, 0.4);
}
[data-theme="dark"] .bs-thumb {
  background: linear-gradient(
    135deg,
    rgba(40, 40, 46, 0.45) 0%,
    rgba(20, 20, 26, 0.6) 100%
  );
}
[data-theme="dark"] .bs-thumb--empty {
  color: rgba(245, 247, 250, 0.45);
}
</style>
