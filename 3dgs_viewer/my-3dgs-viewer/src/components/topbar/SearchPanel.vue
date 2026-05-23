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
  gap: 4px;
  padding: 3px;
  border-radius: 22px;
  background: var(--card-bg, rgba(249, 249, 248, 0.84));
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  box-shadow: 0 10px 26px var(--card-shadow, rgba(0, 0, 0, 0.06));
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}

.search-input {
  flex: 1 1 auto;
  width: auto;
  min-width: 0;
  padding: 10px 14px;
  border: 1px solid var(--input-border, rgba(107, 122, 143, 0.14));
  border-radius: 14px;
  background: var(--input-bg, rgba(255, 255, 255, 0.72));
  outline: none;
  font-size: 13px;
  color: var(--text-primary, #1e1e20);
  font-family: inherit;
}

.search-input:focus {
  border-color: var(--input-focus-border, rgba(107, 122, 143, 0.5));
  box-shadow: 0 0 0 4px var(--input-focus-ring, rgba(107, 122, 143, 0.08));
}

.search-input::placeholder {
  color: var(--text-muted, rgba(30, 30, 32, 0.48));
}

.search-btn {
  flex: 0 0 auto;
  width: 38px;
  height: 38px;
  padding: 0;
  border-radius: 12px;
  border: 1px solid var(--btn-solid-bg, #6b7a8f);
  background: var(--btn-solid-bg, #6b7a8f);
  color: var(--btn-solid-text, #f9f9f8);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  transition: background 180ms ease, transform 180ms ease, box-shadow 180ms ease;
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
  background: var(--btn-solid-hover, #5e6d81);
  border-color: var(--btn-solid-hover, #5e6d81);
  transform: translateY(-1px);
  box-shadow: 0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
}

.search-btn:active {
  transform: translateY(0);
}
</style>
