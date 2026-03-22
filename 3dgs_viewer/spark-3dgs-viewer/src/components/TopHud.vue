<script setup>
defineProps({
  currentFps: {
    type: Number,
    default: 0,
  },
  highlightEnabled: {
    type: Boolean,
    default: true,
  },
  searchQuery: {
    type: String,
    default: '',
  },
  showFocalSettings: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits([
  'update:searchQuery',
  'search',
  'toggle-focal',
  'toggle-highlight',
]);
</script>

<template>
  <div class="hud">
    <div class="search-panel panel-card" @mousedown.stop @touchstart.stop @touchmove.stop @touchend.stop>
      <input
        :value="searchQuery"
        type="text"
        class="search-input"
        placeholder="例如：门口、桌面左侧、正面特写"
        @input="emit('update:searchQuery', $event.target.value)"
        @keyup.enter="emit('search')"
      />
      <button class="panel-btn panel-btn--solid" @click="emit('search')">检索视角</button>
    </div>

    <div class="toolbar">
      <button class="panel-btn panel-btn--ghost" @click="emit('toggle-focal')">
        {{ showFocalSettings ? '收起焦距' : '焦距设置' }}
      </button>
      <button class="panel-btn panel-btn--ghost" @click="emit('toggle-highlight')">
        {{ highlightEnabled ? '关闭特效' : '开启特效' }}
      </button>
      <div class="fps-chip" v-if="currentFps > 0">FPS {{ currentFps }}</div>
    </div>
  </div>
</template>
