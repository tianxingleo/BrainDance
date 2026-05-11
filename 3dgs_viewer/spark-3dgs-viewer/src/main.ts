import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'

// 占位函数：在 Vue 组件加载前提供，确保 Flutter 调用不报错
// 真实实现在 SparkGaussianViewer.vue 的 onMounted 中定义
window.setModelListForTimePeeling = window.setModelListForTimePeeling || function() {}
window.loadModelFromFlutter = window.loadModelFromFlutter || function() {}
window.setThemeFromFlutter = window.setThemeFromFlutter || function() {}

const postBridgeMessage = (payload: Record<string, unknown>) => {
  window.BrainDanceChannel?.postMessage?.(JSON.stringify(payload))
}

window.addEventListener('error', (event) => {
  postBridgeMessage({
    status: 'error',
    msg: event.message || 'WebGL viewer script error',
    source: event.filename,
    line: event.lineno,
    column: event.colno,
  })
})

window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason
  postBridgeMessage({
    status: 'error',
    msg: reason instanceof Error ? reason.message : String(reason),
    source: 'unhandledrejection',
  })
})

createApp(App).mount('#app')
