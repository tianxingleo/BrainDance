<script setup>
</script>

<template>
  <div
    class="topbar"
    @mousedown.stop
    @touchstart.stop
    @touchmove.stop
    @touchend.stop
    @touchcancel.stop
    @wheel.stop
  >
    <div class="topbar-slot topbar-slot--left">
      <slot name="left" />
    </div>
    <div class="topbar-slot topbar-slot--center">
      <slot name="center" />
    </div>
    <div class="topbar-slot topbar-slot--right">
      <slot name="right" />
    </div>
  </div>
</template>

<style scoped>
.topbar {
  position: absolute;
  top: var(--flutter-safe-top, 56px);
  left: var(--flutter-safe-left, 14px);
  right: var(--flutter-safe-right, 14px);
  z-index: 160;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px;
  pointer-events: none;
}

.topbar-slot {
  display: flex;
  align-items: center;
  pointer-events: auto;
}

.topbar-slot--left {
  justify-content: flex-start;
}

.topbar-slot--center {
  min-width: 0;
  justify-content: stretch;
}

.topbar-slot--right {
  justify-content: flex-end;
}
</style>
