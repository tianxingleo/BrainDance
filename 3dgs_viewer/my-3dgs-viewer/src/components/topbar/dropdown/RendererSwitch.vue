<script setup>
const props = defineProps({
  useSpark: { type: Boolean, default: false },
});
const emit = defineEmits(['update:useSpark']);

const select = (next) => {
  if (next === props.useSpark) return;
  emit('update:useSpark', next);
};
</script>

<template>
  <div class="rs-section">
    <div class="rs-label">渲染器</div>
    <div class="rs-segmented" role="tablist">
      <button
        type="button"
        class="rs-chip"
        :class="{ 'rs-chip--active': !props.useSpark }"
        @click="select(false)"
      >
        原版
      </button>
      <button
        type="button"
        class="rs-chip"
        :class="{ 'rs-chip--active': props.useSpark }"
        @click="select(true)"
      >
        Spark
      </button>
    </div>
  </div>
</template>

<style scoped>
.rs-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px 12px;
}

.rs-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--eyebrow-color, #6b7a8f);
}

.rs-segmented {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px;
  border-radius: 14px;
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.08));
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
}

.rs-chip {
  appearance: none;
  flex: 1 1 0;
  border: 0;
  background: transparent;
  color: var(--chip-inactive-text, #6b7280);
  padding: 8px 12px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
  font-family: inherit;
  -webkit-tap-highlight-color: transparent;
}

.rs-chip--active {
  background: var(--chip-active-bg, #1e1e20);
  color: var(--chip-active-text, #f5f4ef);
  box-shadow: 0 6px 14px var(--card-shadow, rgba(0, 0, 0, 0.06));
}

.rs-chip:not(.rs-chip--active):hover {
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.14));
  color: var(--chip-hover-text, #273142);
}
</style>
