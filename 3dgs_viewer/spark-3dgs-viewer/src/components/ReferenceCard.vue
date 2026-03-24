<script setup>
defineProps({
  activeImage: {
    type: String,
    default: '',
  },
  activeTag: {
    type: String,
    default: '',
  },
  sceneMetadata: {
    type: Object,
    default: () => ({}),
  },
});

const emit = defineEmits(['close']);
</script>

<template>
  <div v-if="activeImage" class="reference-card panel-card" @click="emit('close')">
    <div class="eyebrow">Reference Still</div>
    <div class="panel-title">参考原图</div>
    <img :src="activeImage" class="reference-image" />
    <div class="reference-meta" v-if="activeTag">
      <span class="meta-chip meta-chip--accent">{{ activeTag }}</span>
    </div>
    <div class="reference-meta" v-if="sceneMetadata.fl_y">
      <span class="meta-chip">焦距: {{ Number(sceneMetadata.fl_y).toFixed(1) }} px</span>
      <span class="meta-chip">分辨率: {{ sceneMetadata.w }}x{{ sceneMetadata.h }}</span>
    </div>
    <div class="reference-hint">点击关闭对比</div>
  </div>
</template>
