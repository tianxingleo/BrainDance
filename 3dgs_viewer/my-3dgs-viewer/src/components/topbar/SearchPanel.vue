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
    <div class="search-field" :class="{ 'search-field--filled': props.query }">
      <span class="search-leading" aria-hidden="true">
        <svg viewBox="0 0 24 24" focusable="false">
          <path
            d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"
          />
        </svg>
      </span>
      <input
        type="text"
        class="search-input"
        :value="props.query"
        :placeholder="props.placeholder"
        @input="onInput"
        @keyup.enter="onSubmit"
      />
    </div>
    <button
      type="button"
      class="search-btn"
      aria-label="检索视角"
      title="检索视角"
      @click="onSubmit"
    >
      <span class="search-btn-label">搜索</span>
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
  gap: 8px;
}

.search-field {
  position: relative;
  display: flex;
  align-items: center;
  flex: 1 1 auto;
  min-width: 0;
  height: 46px;
  padding: 0 16px 0 8px;
  border-radius: 23px;
  border: 1px solid var(--card-border, rgba(107, 122, 143, 0.16));
  background: var(--card-bg, rgba(249, 249, 248, 0.84));
  box-shadow: 0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  transition:
    border-color 220ms cubic-bezier(0.2, 0.8, 0.2, 1),
    box-shadow 220ms cubic-bezier(0.2, 0.8, 0.2, 1);
}

.search-field:focus-within {
  border-color: var(--input-focus-border, rgba(107, 122, 143, 0.5));
  box-shadow: 0 10px 22px var(--card-shadow, rgba(0, 0, 0, 0.1)),
    0 0 0 4px var(--input-focus-ring, rgba(107, 122, 143, 0.08));
}

.search-leading {
  flex: 0 0 auto;
  width: 34px;
  height: 100%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted, rgba(30, 30, 32, 0.48));
  pointer-events: none;
  transition: color 220ms ease;
}

.search-field:focus-within .search-leading {
  color: var(--text-primary, #1e1e20);
}

.search-leading svg {
  width: 18px;
  height: 18px;
  fill: currentColor;
  display: block;
}

.search-input {
  flex: 1 1 auto;
  width: auto;
  min-width: 0;
  height: 100%;
  padding: 0 4px;
  border: none;
  background: transparent;
  outline: none;
  font-size: 13px;
  color: var(--text-primary, #1e1e20);
  font-family: inherit;
}

.search-input::placeholder {
  color: var(--text-muted, rgba(30, 30, 32, 0.48));
}

.search-btn {
  flex: 0 0 auto;
  height: 46px;
  padding: 0 18px;
  border-radius: 23px;
  border: 1px solid transparent;
  background: var(--btn-solid-bg, #6b7a8f);
  color: var(--btn-solid-text, #f9f9f8);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.2px;
  font-family: inherit;
  -webkit-tap-highlight-color: transparent;
  box-shadow: 0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
  transition:
    background-color 220ms cubic-bezier(0.2, 0.8, 0.2, 1),
    transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1),
    box-shadow 220ms cubic-bezier(0.2, 0.8, 0.2, 1);
  will-change: transform;
}

.search-btn-label {
  display: inline-block;
  line-height: 1;
}

.search-btn:hover {
  background: var(--btn-solid-hover, #5e6d81);
  transform: translateY(-1px);
  box-shadow: 0 12px 22px var(--card-shadow, rgba(0, 0, 0, 0.14));
}

.search-btn:active {
  transform: translateY(0) scale(0.96);
  box-shadow: 0 2px 6px var(--card-shadow, rgba(0, 0, 0, 0.14));
  transition-duration: 90ms;
}

.search-btn:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px var(--input-focus-ring, rgba(107, 122, 143, 0.18)),
    0 8px 18px var(--card-shadow, rgba(0, 0, 0, 0.08));
}
</style>
