<script setup>
const emit = defineEmits(['exit']);
const onClick = () => emit('exit');
</script>

<template>
  <button
    class="exit-btn"
    type="button"
    aria-label="退出"
    title="退出"
    @click="onClick"
  >
    <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
      <path
        d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20z"
      />
    </svg>
  </button>
</template>

<style scoped>
.exit-btn {
  width: 46px;
  height: 46px;
  border-radius: 50%;
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  background: var(--card-bg, rgba(249, 249, 248, 0.84));
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: 0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
  color: var(--text-primary, #1e1e20);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  transition:
    transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1),
    background-color 220ms cubic-bezier(0.2, 0.8, 0.2, 1),
    box-shadow 220ms cubic-bezier(0.2, 0.8, 0.2, 1);
  -webkit-tap-highlight-color: transparent;
  will-change: transform;
}

.exit-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 22px var(--card-shadow, rgba(0, 0, 0, 0.12));
}

.exit-btn:active {
  transform: translateY(0) scale(0.94);
  box-shadow: 0 2px 6px var(--card-shadow, rgba(0, 0, 0, 0.12));
  transition-duration: 90ms;
}

.exit-btn:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px var(--input-focus-ring, rgba(107, 122, 143, 0.18)),
    0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
}

.exit-btn svg {
  width: 18px;
  height: 18px;
  display: block;
  fill: currentColor;
}
</style>
