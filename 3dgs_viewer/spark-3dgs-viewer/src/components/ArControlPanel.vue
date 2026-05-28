<script setup lang="ts">
import type { ArTransform } from '../types/ar'

const props = defineProps<{
  modelValue: ArTransform
}>()

const emit = defineEmits<{
  'update:modelValue': [value: ArTransform]
  reset: []
}>()

const update = (patch: Partial<ArTransform>) => {
  emit('update:modelValue', {
    ...props.modelValue,
    ...patch,
  })
}

const adjustScale = (delta: number) => {
  update({ scale: Math.max(0.01, Number((props.modelValue.scale + delta).toFixed(3))) })
}

const adjustRotationY = (delta: number) => {
  const next = [...props.modelValue.rotation] as [number, number, number]
  next[1] = Number((next[1] + delta).toFixed(3))
  update({ rotation: next })
}

const adjustOffsetY = (delta: number) => {
  const next = [...props.modelValue.offset] as [number, number, number]
  next[1] = Number((next[1] + delta).toFixed(3))
  update({ offset: next })
}
</script>

<template>
  <div class="ar-controls">
    <button type="button" @click="adjustScale(-0.02)">缩小</button>
    <button type="button" @click="adjustScale(0.02)">放大</button>
    <button type="button" @click="adjustRotationY(-0.157)">左转</button>
    <button type="button" @click="adjustRotationY(0.157)">右转</button>
    <button type="button" @click="adjustOffsetY(0.01)">上移</button>
    <button type="button" @click="adjustOffsetY(-0.01)">下移</button>
    <button type="button" class="secondary" @click="emit('reset')">重置</button>
  </div>
</template>

<style scoped>
.ar-controls {
  position: fixed;
  left: 50%;
  bottom: 76px;
  z-index: 20;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  width: min(92vw, 560px);
  transform: translateX(-50%);
  justify-content: center;
}

.ar-controls button {
  border: 0;
  border-radius: 999px;
  padding: 10px 14px;
  color: #fff;
  background: rgba(12, 18, 30, 0.76);
  backdrop-filter: blur(8px);
  font-size: 14px;
}

.ar-controls button.secondary {
  background: rgba(92, 103, 125, 0.78);
}
</style>

