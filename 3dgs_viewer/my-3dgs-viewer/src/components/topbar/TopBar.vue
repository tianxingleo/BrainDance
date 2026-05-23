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
  top: 8px;
  left: 8px;
  right: 8px;
  z-index: 130;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 2px;
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
