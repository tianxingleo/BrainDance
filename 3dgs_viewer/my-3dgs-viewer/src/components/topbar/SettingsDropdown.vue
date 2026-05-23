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
          d="M19.43 12.98c.04-.32.07-.65.07-.98s-.03-.66-.07-.98l2.11-1.65a.5.5 0 0 0 .12-.64l-2-3.46a.5.5 0 0 0-.61-.22l-2.49 1a7.32 7.32 0 0 0-1.69-.98l-.38-2.65A.5.5 0 0 0 14 2h-4a.5.5 0 0 0-.5.42l-.38 2.65a7.49 7.49 0 0 0-1.69.98l-2.49-1a.5.5 0 0 0-.61.22l-2 3.46a.5.5 0 0 0 .12.64L4.57 11.02c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65a.5.5 0 0 0-.12.64l2 3.46a.5.5 0 0 0 .61.22l2.49-1c.52.4 1.08.74 1.69.98l.38 2.65A.5.5 0 0 0 10 22h4a.5.5 0 0 0 .5-.42l.38-2.65c.61-.24 1.17-.58 1.69-.98l2.49 1a.5.5 0 0 0 .61-.22l2-3.46a.5.5 0 0 0-.12-.64L19.43 12.98zM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7z"
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
  transition:
    transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1),
    background-color 220ms cubic-bezier(0.2, 0.8, 0.2, 1),
    box-shadow 220ms cubic-bezier(0.2, 0.8, 0.2, 1);
  -webkit-tap-highlight-color: transparent;
  will-change: transform;
}

.dd-trigger:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 22px var(--card-shadow, rgba(0, 0, 0, 0.12));
}

.dd-trigger:active {
  transform: translateY(0) scale(0.94);
  box-shadow: 0 2px 6px var(--card-shadow, rgba(0, 0, 0, 0.12));
  transition-duration: 90ms;
}

.dd-trigger:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px var(--input-focus-ring, rgba(107, 122, 143, 0.18)),
    0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
}

.dd-trigger--open {
  background: var(--chip-active-bg, #1e1e20);
  color: var(--chip-active-text, #f5f4ef);
  border-color: var(--chip-active-bg, #1e1e20);
}

.dd-trigger svg {
  width: 20px;
  height: 20px;
  display: block;
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
  transform-origin: top right;
  will-change: transform, opacity;
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

.dd-panel-enter-active {
  transition:
    opacity 220ms cubic-bezier(0.2, 0.8, 0.2, 1),
    transform 260ms cubic-bezier(0.2, 0.8, 0.2, 1),
    filter 220ms cubic-bezier(0.2, 0.8, 0.2, 1);
}

.dd-panel-leave-active {
  transition:
    opacity 160ms cubic-bezier(0.4, 0, 1, 1),
    transform 200ms cubic-bezier(0.4, 0, 1, 1),
    filter 160ms cubic-bezier(0.4, 0, 1, 1);
}

.dd-panel-enter-from,
.dd-panel-leave-to {
  opacity: 0;
  transform: translateY(-8px) scale(0.92);
  filter: blur(2px);
}

.dd-panel-enter-to,
.dd-panel-leave-from {
  opacity: 1;
  transform: translateY(0) scale(1);
  filter: blur(0);
}
</style>
