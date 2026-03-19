<script setup>
defineProps({
  focalMax: {
    type: Number,
    required: true,
  },
  focalMin: {
    type: Number,
    required: true,
  },
  manualFocalPx: {
    type: Number,
    default: null,
  },
  currentViewFov: {
    type: Number,
    default: 0,
  },
  currentViewFocalPx: {
    type: Number,
    default: 0,
  },
});

const emit = defineEmits([
  'update:manualFocalPx',
  'input-focal',
  'change-focal',
  'reset-focal',
]);
</script>

<template>
  <div
    class="focal-panel panel-card"
    @mousedown.stop
    @touchstart.stop
    @touchmove.stop
    @touchend.stop
    @touchcancel.stop
  >
    <div class="eyebrow">Lens Control</div>
    <div class="panel-title">镜头焦距</div>
    <input
      :value="manualFocalPx"
      type="range"
      :min="focalMin"
      :max="focalMax"
      step="1"
      @input="emit('update:manualFocalPx', Number($event.target.value)); emit('input-focal')"
    />
    <div class="focal-row">
      <input
        :value="manualFocalPx"
        class="focal-number"
        type="number"
        :min="focalMin"
        :max="focalMax"
        step="1"
        @input="emit('update:manualFocalPx', Number($event.target.value))"
        @change="emit('change-focal')"
      />
      <span>px</span>
    </div>
    <div class="focal-row">
      <span>当前 FOV: {{ currentViewFov.toFixed(1) }}°</span>
    </div>
    <div class="focal-row">
      <span>当前焦距: {{ currentViewFocalPx.toFixed(1) }} px</span>
    </div>
    <button class="panel-btn panel-btn--solid" @click="emit('reset-focal')">恢复拍摄焦距</button>
  </div>
</template>
