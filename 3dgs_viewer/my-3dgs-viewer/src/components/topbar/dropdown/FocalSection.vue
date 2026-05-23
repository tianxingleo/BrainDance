<script setup>
import { ref } from 'vue';

const props = defineProps({
  manualFocalPx: { type: [Number, null], default: null },
  focalMin: { type: Number, required: true },
  focalMax: { type: Number, required: true },
  currentViewFov: { type: Number, default: 0 },
  currentViewFocalPx: { type: Number, default: 0 },
});

const emit = defineEmits([
  'update:manualFocalPx',
  'input',
  'change',
  'reset',
]);

const expanded = ref(false);
const toggle = () => {
  expanded.value = !expanded.value;
};

const onRangeInput = (event) => {
  const v = Number(event.target.value);
  if (!Number.isFinite(v)) return;
  emit('update:manualFocalPx', v);
  emit('input');
};

const onNumberChange = (event) => {
  const v = Number(event.target.value);
  if (!Number.isFinite(v)) return;
  emit('update:manualFocalPx', v);
  emit('change');
};

const onReset = () => emit('reset');
</script>

<template>
  <div class="fs-section">
    <button type="button" class="fs-head" @click="toggle">
      <span class="fs-head-text">
        <span class="fs-eyebrow">Lens Control</span>
        <span class="fs-title">焦距</span>
      </span>
      <span class="fs-chevron" :class="{ 'fs-chevron--open': expanded }" aria-hidden="true">▾</span>
    </button>

    <div v-if="expanded" class="fs-body">
      <input
        class="fs-range"
        type="range"
        :value="props.manualFocalPx ?? props.currentViewFocalPx ?? props.focalMin"
        :min="props.focalMin"
        :max="props.focalMax"
        step="1"
        @input="onRangeInput"
      />

      <div class="fs-row">
        <input
          class="fs-number"
          type="number"
          :value="props.manualFocalPx ?? ''"
          :min="props.focalMin"
          :max="props.focalMax"
          step="1"
          @change="onNumberChange"
        />
        <span class="fs-unit">px</span>
      </div>

      <div class="fs-row fs-row--info">
        <span>当前 FOV</span>
        <span>{{ Number(props.currentViewFov).toFixed(1) }}°</span>
      </div>
      <div class="fs-row fs-row--info">
        <span>当前焦距</span>
        <span>{{ Number(props.currentViewFocalPx).toFixed(1) }} px</span>
      </div>

      <button type="button" class="fs-reset" @click="onReset">恢复拍摄焦距</button>
    </div>
  </div>
</template>

<style scoped>
.fs-section {
  display: flex;
  flex-direction: column;
}

.fs-head {
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

.fs-head:hover {
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.1));
}

.fs-head-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.fs-eyebrow {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--eyebrow-color, #6b7a8f);
}

.fs-title {
  font-size: 14px;
  font-weight: 600;
}

.fs-chevron {
  font-size: 16px;
  color: var(--text-muted, rgba(30, 30, 32, 0.5));
  transition: transform 180ms ease;
  display: inline-flex;
}

.fs-chevron--open {
  transform: rotate(180deg);
}

.fs-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 4px 14px 14px;
}

.fs-range {
  width: 100%;
  accent-color: var(--accent, #6d8260);
}

.fs-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 12px;
  color: var(--text-secondary, rgba(30, 30, 32, 0.72));
}

.fs-row--info {
  font-variant-numeric: tabular-nums;
}

.fs-number {
  flex: 1 1 auto;
  width: 100%;
  border-radius: 10px;
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  padding: 8px 10px;
  background: var(--input-bg, rgba(255, 255, 255, 0.72));
  color: var(--text-primary, #1e1e20);
  font-family: inherit;
  font-size: 13px;
  outline: none;
}

.fs-number:focus {
  border-color: var(--input-focus-border, rgba(107, 122, 143, 0.5));
  box-shadow: 0 0 0 4px var(--input-focus-ring, rgba(107, 122, 143, 0.08));
}

.fs-unit {
  font-size: 12px;
  color: var(--text-muted, rgba(30, 30, 32, 0.5));
  flex-shrink: 0;
}

.fs-reset {
  appearance: none;
  width: 100%;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--btn-solid-bg, #6b7a8f);
  background: var(--btn-solid-bg, #6b7a8f);
  color: var(--btn-solid-text, #f9f9f8);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 180ms ease, transform 180ms ease, box-shadow 180ms ease;
  font-family: inherit;
  -webkit-tap-highlight-color: transparent;
}

.fs-reset:hover {
  background: var(--btn-solid-hover, #5e6d81);
  border-color: var(--btn-solid-hover, #5e6d81);
  transform: translateY(-1px);
}
</style>
