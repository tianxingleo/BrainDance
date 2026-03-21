<script setup>
defineProps({
  currentModelUrl: {
    type: String,
    required: true,
  },
  currentPosesPath: {
    type: String,
    required: true,
  },
  highlightStatus: {
    type: String,
    required: true,
  },
  clipEnabled: {
    type: Boolean,
    default: false,
  },
  clipOffset: {
    type: Number,
    default: 0,
  },
});

const emit = defineEmits(['toggle-clip', 'update:clipOffset']);
</script>

<template>
  <div class="status-ribbon panel-card" @mousedown.stop @touchstart.stop @touchmove.stop @touchend.stop>
    <div class="eyebrow">Spark Prototype</div>
    <div class="status-line">{{ highlightStatus }}</div>
    <div class="status-subline">模型: {{ currentModelUrl }}</div>
    <div class="status-subline">位姿: {{ currentPosesPath }}</div>
    <div class="clip-controls">
      <button class="panel-btn panel-btn--ghost clip-toggle" @click="emit('toggle-clip')">
        {{ clipEnabled ? '关闭剖切' : '开启剖切' }}
      </button>
      <input
        class="clip-slider"
        type="range"
        min="-1"
        max="1"
        step="0.01"
        :value="clipOffset"
        @input="emit('update:clipOffset', Number($event.target.value))"
      />
    </div>
  </div>
</template>
