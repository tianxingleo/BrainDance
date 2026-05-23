<script setup>
import { ref } from 'vue';

const props = defineProps({
  speed: { type: Number, required: true },
  progress: { type: Number, required: true },
  loop: { type: Boolean, required: true },
  smoothness: { type: Number, required: true },
  subjectLock: { type: Boolean, required: true },
  isPlaying: { type: Boolean, required: true },
  isPaused: { type: Boolean, required: true },
  canPlay: { type: Boolean, required: true },
  buttonLabel: { type: String, required: true },
});

const emit = defineEmits([
  'update:speed',
  'update:loop',
  'update:smoothness',
  'update:subjectLock',
  'speed-change',
  'style-change',
  'play-toggle',
  'stop',
]);

const expanded = ref(false);
const toggle = () => {
  if (!props.canPlay) return;
  expanded.value = !expanded.value;
};

const onSpeedInput = (event) => {
  const v = Number(event.target.value);
  if (!Number.isFinite(v)) return;
  emit('update:speed', v);
  emit('speed-change');
};

const onSmoothInput = (event) => {
  const v = Number(event.target.value);
  if (!Number.isFinite(v)) return;
  emit('update:smoothness', v);
  emit('style-change');
};

const onLoopChange = (event) => {
  emit('update:loop', !!event.target.checked);
};

const onSubjectLockChange = (event) => {
  emit('update:subjectLock', !!event.target.checked);
  emit('style-change');
};

const stopDisabled = () =>
  !props.isPlaying && !props.isPaused && props.progress === 0;
</script>

<template>
  <div class="cs-section" :class="{ 'cs-section--disabled': !props.canPlay }">
    <button
      type="button"
      class="cs-head"
      :disabled="!props.canPlay"
      @click="toggle"
    >
      <span class="cs-head-text">
        <span class="cs-eyebrow">Camera Move</span>
        <span class="cs-title">运镜</span>
      </span>
      <span class="cs-head-state">
        <span v-if="!props.canPlay" class="cs-na">该模型暂无足够视角</span>
        <span v-else class="cs-chevron" :class="{ 'cs-chevron--open': expanded }" aria-hidden="true">▾</span>
      </span>
    </button>

    <div v-if="expanded && props.canPlay" class="cs-body">
      <div class="cs-actions">
        <button
          type="button"
          class="cs-primary"
          @click="emit('play-toggle')"
        >
          {{ props.buttonLabel }}
        </button>
        <button
          type="button"
          class="cs-secondary"
          :disabled="stopDisabled()"
          @click="emit('stop')"
        >
          停止
        </button>
      </div>

      <div class="cs-row">
        <span>进度</span>
        <span>{{ Math.round(props.progress * 100) }}%</span>
      </div>
      <input
        class="cs-range"
        type="range"
        :value="props.progress * 100"
        min="0"
        max="100"
        step="1"
        disabled
      />

      <div class="cs-row">
        <span>速度</span>
        <span>{{ props.speed.toFixed(2) }}x</span>
      </div>
      <input
        class="cs-range"
        type="range"
        :value="props.speed"
        min="0.25"
        max="3"
        step="0.05"
        @input="onSpeedInput"
      />

      <div class="cs-row">
        <span>平滑</span>
        <span>{{ Math.round(props.smoothness * 100) }}%</span>
      </div>
      <input
        class="cs-range"
        type="range"
        :value="props.smoothness"
        min="0"
        max="1"
        step="0.05"
        @input="onSmoothInput"
      />

      <div class="cs-toggles">
        <label class="cs-check">
          <input type="checkbox" :checked="props.loop" @change="onLoopChange" />
          <span>循环</span>
        </label>
        <label class="cs-check">
          <input type="checkbox" :checked="props.subjectLock" @change="onSubjectLockChange" />
          <span>主体锁定</span>
        </label>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cs-section {
  display: flex;
  flex-direction: column;
}

.cs-section--disabled .cs-head {
  cursor: not-allowed;
  opacity: 0.55;
}

.cs-head {
  appearance: none;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 14px;
  border: 0;
  background: transparent;
  color: var(--text-primary, #1e1e20);
  text-align: left;
  cursor: pointer;
  border-radius: 14px;
  font-family: inherit;
  transition: background 160ms ease;
}

.cs-head:hover:not(:disabled) {
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.1));
}

.cs-head-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.cs-eyebrow {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--eyebrow-color, #6b7a8f);
}

.cs-title {
  font-size: 14px;
  font-weight: 600;
}

.cs-na {
  font-size: 11px;
  color: var(--text-muted, rgba(30, 30, 32, 0.5));
}

.cs-chevron {
  font-size: 16px;
  color: var(--text-muted, rgba(30, 30, 32, 0.5));
  transition: transform 180ms ease;
  display: inline-flex;
}

.cs-chevron--open {
  transform: rotate(180deg);
}

.cs-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 4px 14px 14px;
}

.cs-actions {
  display: flex;
  gap: 8px;
}

.cs-primary,
.cs-secondary {
  appearance: none;
  flex: 1 1 0;
  padding: 10px 12px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 180ms ease, transform 180ms ease, box-shadow 180ms ease;
  font-family: inherit;
  -webkit-tap-highlight-color: transparent;
}

.cs-primary {
  background: var(--btn-solid-bg, #6b7a8f);
  border: 1px solid var(--btn-solid-bg, #6b7a8f);
  color: var(--btn-solid-text, #f9f9f8);
}

.cs-primary:hover {
  background: var(--btn-solid-hover, #5e6d81);
  border-color: var(--btn-solid-hover, #5e6d81);
  transform: translateY(-1px);
}

.cs-secondary {
  background: var(--btn-ghost-bg, rgba(249, 249, 248, 0.84));
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  color: var(--text-primary, #1e1e20);
}

.cs-secondary:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 14px var(--card-shadow, rgba(0, 0, 0, 0.08));
}

.cs-secondary:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.cs-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  color: var(--text-secondary, rgba(30, 30, 32, 0.72));
}

.cs-range {
  width: 100%;
  accent-color: var(--accent, #6d8260);
}

.cs-toggles {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}

.cs-check {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-secondary, rgba(30, 30, 32, 0.72));
}

.cs-check input {
  accent-color: var(--accent, #6d8260);
}
</style>
