import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'

// 占位函数：在 Vue 组件加载前提供，确保 Flutter 调用不报错
// 真实实现在 SparkGaussianViewer.vue 的 onMounted 中定义
window.setModelListForTimePeeling = window.setModelListForTimePeeling || function() {}
window.loadModelFromFlutter = window.loadModelFromFlutter || function() {}

createApp(App).mount('#app')
