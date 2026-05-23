<script setup>
const props = defineProps({
  query: { type: String, default: '' },
  placeholder: { type: String, default: '例如：门口、桌面左侧、正面特写' },
});

const emit = defineEmits(['update:query', 'search']);

const onInput = (event) => {
  emit('update:query', event.target.value);
};

const onSubmit = () => emit('search');
</script>

<template>
  <div class="search-panel" role="search">
    <input
      type="text"
      class="search-input"
      :value="props.query"
      :placeholder="props.placeholder"
      @input="onInput"
      @keyup.enter="onSubmit"
    />
    <button
      type="button"
      class="search-btn"
      aria-label="检索视角"
      title="检索视角"
      @click="onSubmit"
    >
      <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
        <path
          d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"
        />
      </svg>
    </button>
  </div>
</template>

<style scoped>
.search-panel {
  width: 100%;
  display: flex;
  flex: 1 1 auto;
  min-width: 0;
  flex-direction: row;
  align-items: center;
  gap: 6px;
}

.search-input {
  flex: 1 1 auto;
  width: auto;
  min-width: 0;
  height: 46px;
  padding: 0 18px;
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  border-radius: 23px;
  background: var(--card-bg, rgba(249, 249, 248, 0.84));
  box-shadow: 0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  outline: none;
  font-size: 13px;
  color: var(--text-primary, #1e1e20);
  font-family: inherit;
  transition: border-color 180ms ease, box-shadow 180ms ease;
}

.search-input:focus {
  border-color: var(--input-focus-border, rgba(107, 122, 143, 0.5));
  box-shadow: 0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08)),
    0 0 0 4px var(--input-focus-ring, rgba(107, 122, 143, 0.08));
}

.search-input::placeholder {
  color: var(--text-muted, rgba(30, 30, 32, 0.48));
}

.search-btn {
  flex: 0 0 auto;
  width: 46px;
  height: 46px;
  padding: 0;
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
  white-space: nowrap;
  transition: transform 180ms ease, background-color 180ms ease, box-shadow 180ms ease;
  font-family: inherit;
  -webkit-tap-highlight-color: transparent;
}

.search-btn svg {
  width: 18px;
  height: 18px;
  display: block;
  fill: currentColor;
}

.search-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 22px var(--card-shadow, rgba(0, 0, 0, 0.12));
}

.search-btn:active {
  transform: translateY(0);
}
</style>
