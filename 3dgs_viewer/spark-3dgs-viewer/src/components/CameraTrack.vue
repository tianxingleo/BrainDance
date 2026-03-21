<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
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
const isExpanded = ref(false);

const visiblePoses = computed(() => props.filteredPoses.slice(0, 8));
</script>

<template>
  <div
    v-if="props.filteredPoses.length > 0"
    class="camera-track-dock"
  >
    <button
      class="camera-track-toggle panel-btn panel-btn--ghost"
      @click.stop="isExpanded = !isExpanded"
      @mousedown.stop
      @touchstart.stop
      @touchmove.stop
      @touchend.stop
    >
      {{ isExpanded ? '收起自动运镜' : '自动运镜' }}
    </button>

    <div
      v-if="isExpanded"
      class="camera-track panel-card"
      @mousedown.stop
      @touchstart.stop
      @touchmove.stop
      @touchend.stop
    >
      <div class="track-copy">
        <div class="eyebrow">Auto Camera</div>
        <div class="track-text">{{ props.searchQuery ? '按当前检索结果排序' : '优先显示已打标签镜头' }}</div>
      </div>
      <div
        v-for="pose in visiblePoses"
        :key="pose.id || pose.image_url || JSON.stringify(pose.matrix)"
        class="camera-item"
        :class="{ active: props.activeImage === pose.image_url }"
        @click.stop="emit('select-pose', pose)"
      >
        <img v-if="pose.image_url" :src="pose.image_url" class="camera-thumb" />
        <div v-if="pose.tag" class="camera-tag">{{ pose.tag }}</div>
        <span v-else-if="!pose.image_url">未命名视角</span>
      </div>
    </div>
  </div>
</template>
