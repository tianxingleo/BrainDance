<script setup>
const props = defineProps({
  disabled: { type: Boolean, default: false },
});

const emit = defineEmits(['enter-ar']);

const onClick = () => {
  if (props.disabled) return;
  emit('enter-ar');
};
</script>

<template>
  <button
    type="button"
    class="ar-row"
    :disabled="props.disabled"
    @click="onClick"
  >
    <span class="ar-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" focusable="false">
        <path
          d="M12 3 4 7v10l8 4 8-4V7l-8-4zm0 2.2 6 3v.5l-6 3-6-3V8.2l6-3zM6 10.2l5 2.5v6.6l-5-2.5v-6.6zm12 0v6.6l-5 2.5v-6.6l5-2.5z"
        />
      </svg>
    </span>
    <span class="ar-text">
      <span class="ar-title">AR</span>
    </span>
    <span class="ar-chevron" aria-hidden="true">›</span>
  </button>
</template>

<style scoped>
.ar-row {
  appearance: none;
  width: 100%;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border: 0;
  background: transparent;
  color: var(--text-primary, #1e1e20);
  text-align: left;
  cursor: pointer;
  border-radius: 14px;
  transition: background 160ms ease;
  font-family: inherit;
}

.ar-row:hover {
  background: var(--chip-hover-bg, rgba(107, 122, 143, 0.1));
}

.ar-row:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.ar-icon {
  width: 36px;
  height: 36px;
  border-radius: 12px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: var(--chip-active-bg, #1e1e20);
  color: var(--chip-active-text, #f5f4ef);
  flex-shrink: 0;
}

.ar-icon svg {
  width: 20px;
  height: 20px;
  fill: currentColor;
}

.ar-text {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.ar-title {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.2;
}

.ar-chevron {
  font-size: 22px;
  line-height: 1;
  color: var(--text-muted, rgba(30, 30, 32, 0.4));
  flex-shrink: 0;
}
</style>
