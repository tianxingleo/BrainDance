<script setup>
defineProps({
  activeImage: {
    type: String,
    default: '',
  },
  filteredPoses: {
    type: Array,
    default: () => [],
  },
  searchQuery: {
    type: String,
    default: '',
  },
});

const emit = defineEmits(['select-pose']);
</script>

<template>
  <div
    v-if="filteredPoses.length > 0"
    class="camera-track panel-card"
    @mousedown.stop
    @touchstart.stop
    @touchmove.stop
    @touchend.stop
  >
    <div class="track-copy">
      <div class="eyebrow">Shot Strip</div>
      <div class="track-text">{{ searchQuery ? '按当前检索结果排序' : '优先显示已打标签镜头' }}</div>
    </div>
    <div
      v-for="pose in filteredPoses"
      :key="pose.id || pose.image_url || JSON.stringify(pose.matrix)"
      class="camera-item"
      :class="{ active: activeImage === pose.image_url }"
      @click.stop="emit('select-pose', pose)"
    >
      <img v-if="pose.image_url" :src="pose.image_url" class="camera-thumb" />
      <div v-if="pose.tag" class="camera-tag">{{ pose.tag }}</div>
      <span v-else-if="!pose.image_url">未命名视角</span>
    </div>
  </div>
</template>
