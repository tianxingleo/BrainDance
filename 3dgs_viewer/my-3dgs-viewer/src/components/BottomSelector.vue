<script setup>
import { computed, ref, watch, nextTick, onMounted, onBeforeUnmount, shallowRef } from 'vue';

import TimelineElastic from './TimelineElastic.vue';

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

// 拖拽滚动状态
let isDragging = false;
let dragStartX = 0;
let scrollStartLeft = 0;
let dragMoved = false;

const modelItems = shallowRef([]);

// 只要有时间维度数据，就保留切换入口；单个模型也要能进入时间视图。
const showTabs = computed(() => props.models.length > 0);

// 自动切换到可用 tab（仅当模型 tab 不可用时切换，视角为空时保留空状态）
watch([() => props.hasModels, () => props.hasPoses], () => {
  if (mode.value === 'model' && !props.hasModels) mode.value = 'pose';
}, { immediate: true });

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

// 拖拽滚动
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
    <!-- 切换按钮 -->
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

    <!-- 缩略图滚动区 -->
    <div class="bs-track-wrap">
      <div class="bs-track"
        ref="scrollRef"
        @mousedown="onPointerDown"
        @mousemove="onPointerMove"
        @touchstart="onPointerDown"
        @touchmove="onPointerMove"
      >
        <!-- 视角模式 -->
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
            <!-- 文字标签已隐藏，只显示图片 -->
            <!-- <div v-if="pose.tag" class="bs-tag">{{ pose.tag }}</div> -->
          </div>
        </template>

        <!-- 模型模式 (Timeline) -->
        <template v-if="mode === 'model'">
          <TimelineElastic
            :items="modelItems"
            :active-id="activeModelId"
            @select="onClickModel"
          />
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bs-root {
  position: absolute;
  bottom: 32px;
  left: 16px;
  right: 16px;
  z-index: 100;
  display: flex;
  align-items: center;
  gap: 0;
  pointer-events: auto;
  background: var(--card-bg, rgba(249, 249, 248, 0.88));
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-radius: 20px;
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  box-shadow: 0 8px 24px var(--card-shadow, rgba(0, 0, 0, 0.1));
  padding: 10px 0 10px 12px;
}

/* 切换按钮 */
.bs-tabs {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex-shrink: 0;
  padding-right: 10px;
}

.bs-tab {
  padding: 6px 12px;
  border: none;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.1));
  color: var(--text-muted, rgba(30, 30, 32, 0.5));
  white-space: nowrap;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  outline: none;
}
.bs-tab--active {
  background: rgba(204, 154, 92, 0.88);
  color: #fff;
  box-shadow: 0 2px 10px rgba(204, 154, 92, 0.3);
}
.bs-tab:hover:not(.bs-tab--active) {
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.18));
  color: var(--text-secondary, rgba(30, 30, 32, 0.75));
}

/* 缩略图滚动区 */
.bs-track-wrap {
  position: relative;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  -webkit-mask-image: linear-gradient(to right, transparent, black 24px, black calc(100% - 24px), transparent);
  mask-image: linear-gradient(to right, transparent, black 24px, black calc(100% - 24px), transparent);
}

.bs-track {
  display: flex;
  gap: 12px;
  align-items: center;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 6px 36px;
  scrollbar-width: none;
  -ms-overflow-style: none;
  cursor: grab;
  user-select: none;
}
.bs-track::-webkit-scrollbar {
  display: none;
}
.bs-track:active {
  cursor: grabbing;
}

/* 缩略图项 */
.bs-item {
  position: relative;
  width: 96px;
  height: 68px;
  flex-shrink: 0;
  border-radius: 12px;
  overflow: hidden;
  cursor: pointer;
  transition:
    transform 0.25s cubic-bezier(0.22, 1, 0.36, 1),
    opacity 0.25s cubic-bezier(0.22, 1, 0.36, 1),
    border-color 0.25s ease,
    box-shadow 0.25s ease;
  border: 2px solid var(--card-border, rgba(107, 122, 143, 0.12));
  opacity: 0.7;
  box-shadow: 0 2px 8px var(--card-shadow, rgba(0, 0, 0, 0.08));
  transform: scale(0.84);
  transform-origin: center;
  will-change: transform, opacity;
}
.bs-item--active {
  border-color: #CC9A5C;
  opacity: 1;
  box-shadow: 0 4px 16px rgba(204, 154, 92, 0.3);
  transform: scale(1);
}
.bs-item:hover:not(.bs-item--active) {
  opacity: 0.88;
  border-color: var(--input-focus-border, rgba(107, 122, 143, 0.25));
}

.bs-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
  background: rgba(30, 30, 32, 0.5);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  opacity: 0;
  transition: opacity 0.6s cubic-bezier(0.16, 1, 0.3, 1), transform 0.6s cubic-bezier(0.16, 1, 0.3, 1), filter 0.6s cubic-bezier(0.16, 1, 0.3, 1);
  transform: scale(0.95);
  filter: blur(4px);
}
.bs-thumb.bs-thumb--loaded {
  opacity: 1;
  transform: scale(1);
  filter: blur(0px);
}
.bs-thumb--empty {
  opacity: 1;
  transform: scale(1);
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(255, 255, 255, 0.4);
  font-size: 11px;
}

/* 视角标签 */
.bs-tag {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 2px 6px;
  font-size: 9px;
  font-weight: 600;
  color: #fff;
  background: linear-gradient(transparent, rgba(0,0,0,0.6));
  text-align: center;
  line-height: 1.4;
  pointer-events: none;
}

/* 模型时间标签 */
.bs-time {
  position: absolute;
  bottom: 1px;
  left: 0;
  right: 0;
  text-align: center;
  font-size: 8px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.85);
  text-shadow: 0 1px 3px rgba(0,0,0,0.7);
  pointer-events: none;
  line-height: 1.2;
}

</style>
