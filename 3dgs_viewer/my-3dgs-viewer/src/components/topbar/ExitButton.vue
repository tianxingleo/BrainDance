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
        d="M14 4h-2.5L7 8.5V11H4v2h3v2.5L11.5 20H14v-2h-1.5L9 14.5V13h7v-2H9V9.5L12.5 6H14V4z"
        opacity="0"
      />
      <path
        d="M10.6 6.4 8.5 4.3 0.8 12l7.7 7.7 2.1-2.1L6.5 13.5H22v-3H6.5l4.1-4.1z"
        transform="translate(1 0)"
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
  transition: transform 180ms ease, background-color 180ms ease, box-shadow 180ms ease;
  -webkit-tap-highlight-color: transparent;
}

.exit-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 22px var(--card-shadow, rgba(0, 0, 0, 0.12));
}

.exit-btn:active {
  transform: translateY(0);
}

.exit-btn svg {
  width: 18px;
  height: 18px;
  fill: currentColor;
}
</style>
