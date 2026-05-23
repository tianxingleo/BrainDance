<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';
import ArEntryItem from './dropdown/ArEntryItem.vue';
import ViewModeToggle from './dropdown/ViewModeToggle.vue';
import CinematicSection from './dropdown/CinematicSection.vue';
import FocalSection from './dropdown/FocalSection.vue';

const props = defineProps({
  // 视角模式
  viewMode: { type: String, default: 'free' },
  // 运镜
  cinematicSpeed: { type: Number, required: true },
  cinematicProgress: { type: Number, required: true },
  cinematicLoop: { type: Boolean, required: true },
  cinematicSmoothness: { type: Number, required: true },
  cinematicSubjectLock: { type: Boolean, required: true },
  isCinematicPlaying: { type: Boolean, required: true },
  isCinematicPaused: { type: Boolean, required: true },
  canPlayCinematic: { type: Boolean, required: true },
  cinematicButtonLabel: { type: String, required: true },
  // 焦距
  manualFocalPx: { type: [Number, null], default: null },
  focalMin: { type: Number, required: true },
  focalMax: { type: Number, required: true },
  currentViewFov: { type: Number, default: 0 },
  currentViewFocalPx: { type: Number, default: 0 },
  // AR
  arDisabled: { type: Boolean, default: false },
});

const emit = defineEmits([
  'enter-ar',
  'update:viewMode',
  'update:cinematicSpeed',
  'update:cinematicLoop',
  'update:cinematicSmoothness',
  'update:cinematicSubjectLock',
  'cinematic-speed-change',
  'cinematic-style-change',
  'cinematic-play-toggle',
  'cinematic-stop',
  'update:manualFocalPx',
  'focal-input',
  'focal-change',
  'focal-reset',
]);

const open = ref(false);
const rootRef = ref(null);

const toggle = () => {
  open.value = !open.value;
};

const close = () => {
  open.value = false;
};

const onDocumentPointerDown = (event) => {
  if (!open.value) return;
  const root = rootRef.value;
  if (root && !root.contains(event.target)) {
    open.value = false;
  }
};

const onKeydown = (event) => {
  if (event.key === 'Escape') close();
};

onMounted(() => {
  document.addEventListener('pointerdown', onDocumentPointerDown, true);
  document.addEventListener('keydown', onKeydown);
});

onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', onDocumentPointerDown, true);
  document.removeEventListener('keydown', onKeydown);
});

const onEnterAr = () => {
  close();
  emit('enter-ar');
};
</script>

<template>
  <div ref="rootRef" class="dd-root">
    <button
      type="button"
      class="dd-trigger"
      :class="{ 'dd-trigger--open': open }"
      :aria-expanded="open"
      aria-haspopup="true"
      aria-label="设置菜单"
      title="设置"
      @click="toggle"
    >
      <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
        <path
          d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.61-.22l-2.39.96a7.05 7.05 0 0 0-1.62-.94l-.36-2.54a.5.5 0 0 0-.5-.42h-3.84a.5.5 0 0 0-.5.42l-.36 2.54c-.59.24-1.13.55-1.62.94l-2.39-.96a.5.5 0 0 0-.61.22L2.65 8.84a.5.5 0 0 0 .12.64l2.03 1.58c-.04.31-.06.63-.06.94 0 .31.02.63.06.94l-2.03 1.58a.5.5 0 0 0-.12.64l1.92 3.32c.14.24.43.34.69.22l2.39-.96c.49.39 1.03.7 1.62.94l.36 2.54c.05.24.26.42.5.42h3.84c.24 0 .45-.18.5-.42l.36-2.54c.59-.24 1.13-.55 1.62-.94l2.39.96c.26.12.55.02.69-.22l1.92-3.32a.5.5 0 0 0-.12-.64l-2.03-1.58zM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7z"
        />
      </svg>
    </button>

    <transition name="dd-panel">
      <div v-if="open" class="dd-panel" role="menu">
        <div class="dd-section dd-section--ar">
          <ArEntryItem :disabled="props.arDisabled" @enter-ar="onEnterAr" />
        </div>

        <div class="dd-divider" />

        <div class="dd-section">
          <ViewModeToggle
            :mode="props.viewMode"
            @update:mode="(v) => emit('update:viewMode', v)"
          />
        </div>

        <div class="dd-divider" />

        <div class="dd-section">
          <CinematicSection
            :speed="props.cinematicSpeed"
            :progress="props.cinematicProgress"
            :loop="props.cinematicLoop"
            :smoothness="props.cinematicSmoothness"
            :subject-lock="props.cinematicSubjectLock"
            :is-playing="props.isCinematicPlaying"
            :is-paused="props.isCinematicPaused"
            :can-play="props.canPlayCinematic"
            :button-label="props.cinematicButtonLabel"
            @update:speed="(v) => emit('update:cinematicSpeed', v)"
            @update:loop="(v) => emit('update:cinematicLoop', v)"
            @update:smoothness="(v) => emit('update:cinematicSmoothness', v)"
            @update:subjectLock="(v) => emit('update:cinematicSubjectLock', v)"
            @speed-change="emit('cinematic-speed-change')"
            @style-change="emit('cinematic-style-change')"
            @play-toggle="emit('cinematic-play-toggle')"
            @stop="emit('cinematic-stop')"
          />
        </div>

        <div class="dd-divider" />

        <div class="dd-section">
          <FocalSection
            :manual-focal-px="props.manualFocalPx"
            :focal-min="props.focalMin"
            :focal-max="props.focalMax"
            :current-view-fov="props.currentViewFov"
            :current-view-focal-px="props.currentViewFocalPx"
            @update:manualFocalPx="(v) => emit('update:manualFocalPx', v)"
            @input="emit('focal-input')"
            @change="emit('focal-change')"
            @reset="emit('focal-reset')"
          />
        </div>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.dd-root {
  position: relative;
  display: inline-flex;
}

.dd-trigger {
  appearance: none;
  width: 46px;
  height: 46px;
  border-radius: 50%;
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  background: var(--card-bg, rgba(249, 249, 248, 0.84));
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: 0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
  color: var(--text-primary, #1e1e20);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  transition: transform 180ms ease, background-color 180ms ease, box-shadow 180ms ease;
  -webkit-tap-highlight-color: transparent;
}

.dd-trigger:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 22px var(--card-shadow, rgba(0, 0, 0, 0.12));
}

.dd-trigger--open {
  background: var(--chip-active-bg, #1e1e20);
  color: var(--chip-active-text, #f5f4ef);
  border-color: var(--chip-active-bg, #1e1e20);
}

.dd-trigger svg {
  width: 20px;
  height: 20px;
  fill: currentColor;
}

.dd-panel {
  position: absolute;
  top: calc(100% + 10px);
  right: 0;
  width: min(86vw, 320px);
  background: var(--card-bg, rgba(249, 249, 248, 0.94));
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  border-radius: 22px;
  box-shadow: 0 20px 38px var(--card-shadow, rgba(0, 0, 0, 0.12));
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  padding: 6px;
  z-index: 200;
  max-height: calc(100vh - var(--flutter-safe-top, 92px) - 80px);
  overflow-y: auto;
  overflow-x: hidden;
}

.dd-section {
  padding: 0;
}

.dd-section--ar {
  padding: 4px;
}

.dd-divider {
  height: 1px;
  background: var(--card-border, rgba(107, 122, 143, 0.14));
  margin: 4px 8px;
}

.dd-panel-enter-active,
.dd-panel-leave-active {
  transition: opacity 160ms ease, transform 160ms ease;
}

.dd-panel-enter-from,
.dd-panel-leave-to {
  opacity: 0;
  transform: translateY(-6px) scale(0.98);
}
</style>
