<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import type { RealtimeChannel } from '@supabase/supabase-js'
import { Icon } from '@iconify/vue'
import dayjs from 'dayjs'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, PieChart, BarChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { supabase } from './lib/supabase'

use([CanvasRenderer, LineChart, PieChart, BarChart, GridComponent, TooltipComponent, LegendComponent])

type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | string

interface ProcessingTask {
  id: string
  display_name: string | null
  scene_id: string
  user_id: string
  status: TaskStatus
  task_type: string | null
  quality_score: number | null
  created_at: string
  updated_at: string
  logs: unknown
}

interface BucketStat {
  id: string
  public: boolean
  objectCount: number
  totalBytes: number
  latestUpdatedAt: string | null
}

interface StorageListObject {
  name: string
  id: string | null
  updated_at: string | null
  metadata?: {
    size?: number
  } | null
}

interface BucketProbeResult {
  stat: BucketStat
  ok: boolean
}

interface EdgeFunctionCheck {
  name: string
  status: 'ok' | 'missing' | 'error'
  httpStatus: number | null
  latencyMs: number | null
  lastCheckedAt: string | null
  message: string
}

const loading = ref(true)
const refreshing = ref(false)
const storageLoading = ref(false)
const errorMessage = ref('')
const storageError = ref('')
const lastUpdated = ref<string | null>(null)

const taskRows = ref<ProcessingTask[]>([])
const modelAssetCount = ref(0)
const memoryPoseCount = ref(0)

const selectedStatus = ref<'all' | TaskStatus>('all')
const selectedTaskType = ref('all')
const searchKeyword = ref('')
const autoRefresh = ref(true)
const refreshSeconds = ref(60)
const taskTrendRange = ref<'24h' | '7d' | '30d' | 'all'>('all')
const isDarkTheme = ref(true)
const accentColor = ref('#71839a')

const THEME_STORAGE_KEY = 'dashboard-theme-dark'
const ACCENT_STORAGE_KEY = 'dashboard-theme-accent'

const channelState = ref<Record<string, string>>({
  processing_tasks: 'connecting',
  model_assets: 'connecting',
  memory_poses: 'connecting',
})

const storageStats = ref<BucketStat[]>([])
const storageApiReachable = ref(false)
const storageProbeMode = ref<'bucket_list' | 'known_buckets' | 'none'>('none')
const edgeChecks = ref<EdgeFunctionCheck[]>([])
const edgeLoading = ref(false)

const dbCounts = ref<Record<string, number>>({
  processing_tasks: 0,
  model_assets: 0,
  memory_poses: 0,
  rag_docs: 0,
  tasks: 0,
})

const timeBasedStats = ref({
  tasks24h: 0,
  failed24h: 0,
  completed24h: 0,
  activeUsers7d: 0,
  assets7d: 0,
})

const statusMap: Record<string, { label: string; type: 'success' | 'warning' | 'danger' | 'info' }> = {
  pending: { label: '排队中', type: 'warning' },
  processing: { label: '处理中', type: 'info' },
  completed: { label: '已完成', type: 'success' },
  failed: { label: '失败', type: 'danger' },
}

const pendingCount = computed(() => taskRows.value.filter((item) => item.status === 'pending').length)
const processingCount = computed(() => taskRows.value.filter((item) => item.status === 'processing').length)
const completedCount = computed(() => taskRows.value.filter((item) => item.status === 'completed').length)
const failedCount = computed(() => taskRows.value.filter((item) => item.status === 'failed').length)

const successRate = computed(() => {
  const doneTotal = completedCount.value + failedCount.value
  if (!doneTotal) return 0
  return Math.round((completedCount.value / doneTotal) * 100)
})

const avgQualityScore = computed(() => {
  const scores = taskRows.value
    .map((item) => item.quality_score)
    .filter((score): score is number => typeof score === 'number')

  if (!scores.length) return '-'

  const total = scores.reduce((sum, score) => sum + score, 0)
  return (total / scores.length).toFixed(1)
})

const avgProcessMinutes = computed(() => {
  const completedTasks = taskRows.value.filter((item) => item.status === 'completed')
  if (!completedTasks.length) return '-'

  const durations = completedTasks
    .map((item) => {
      const created = dayjs(item.created_at)
      const updated = dayjs(item.updated_at)
      return Math.max(updated.diff(created, 'second'), 0)
    })
    .filter((s) => Number.isFinite(s))

  if (!durations.length) return '-'

  const avg = durations.reduce((sum, s) => sum + s, 0) / durations.length
  return `${(avg / 60).toFixed(1)} min`
})

const latestTaskUpdatedAt = computed(() => {
  if (!taskRows.value.length) return null
  return taskRows.value.map((item) => dayjs(item.updated_at)).sort((a, b) => b.valueOf() - a.valueOf())[0]
})

const taskFreshnessText = computed(() => {
  if (!latestTaskUpdatedAt.value) return '暂无任务活动'
  const diffSeconds = dayjs().diff(latestTaskUpdatedAt.value, 'second')
  if (diffSeconds < 60) return `${diffSeconds}s 前更新`
  if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m 前更新`
  return `${Math.floor(diffSeconds / 3600)}h 前更新`
})

const workerOnline = computed(() => {
  if (!latestTaskUpdatedAt.value) return false
  return dayjs().diff(latestTaskUpdatedAt.value, 'second') <= 180
})

const realtimeHealthy = computed(() =>
  Object.values(channelState.value).every((state) => state === 'SUBSCRIBED'),
)

const realtimeStatusText = computed(() => (realtimeHealthy.value ? '已连接' : '连接中'))

const taskTypeOptions = computed(() => {
  const set = new Set<string>()
  taskRows.value.forEach((item) => {
    if (item.task_type) set.add(item.task_type)
  })
  return Array.from(set).sort()
})

const filteredTasks = computed(() => {
  const keyword = searchKeyword.value.trim().toLowerCase()

  return taskRows.value.filter((item) => {
    if (selectedStatus.value !== 'all' && item.status !== selectedStatus.value) return false
    if (selectedTaskType.value !== 'all' && item.task_type !== selectedTaskType.value) return false
    if (!keyword) return true

    const joined = `${item.display_name || ''} ${item.scene_id} ${item.user_id} ${item.id}`.toLowerCase()
    return joined.includes(keyword)
  })
})

const taskQueue = computed(() => filteredTasks.value.slice(0, 20))
const failedTasks = computed(() => taskRows.value.filter((item) => item.status === 'failed').slice(0, 6))

const totalStorageBytes = computed(() =>
  storageStats.value.reduce((sum, item) => sum + item.totalBytes, 0),
)

const edgeHealthyCount = computed(() => edgeChecks.value.filter((item) => item.status === 'ok').length)
const queueCount = computed(() => pendingCount.value + processingCount.value)

const refreshModeText = computed(() => (autoRefresh.value ? `${refreshSeconds.value}s 自动刷新` : '手动刷新'))

const storageModeText = computed(() => {
  if (storageProbeMode.value === 'bucket_list') return '列桶模式'
  if (storageProbeMode.value === 'known_buckets') return '已知桶探测'
  return '不可用'
})

type MetricTone = 'good' | 'warn' | 'bad' | 'neutral'

const successSeverity = computed<MetricTone>(() => {
  if (successRate.value < 70) return 'bad'
  if (successRate.value < 90) return 'warn'
  return 'good'
})

const failureSeverity = computed<MetricTone>(() => {
  if (failedCount.value >= 5) return 'bad'
  if (failedCount.value > 0) return 'warn'
  return 'good'
})

const queueSeverity = computed<MetricTone>(() => {
  if (queueCount.value >= 30) return 'bad'
  if (queueCount.value >= 10) return 'warn'
  return 'good'
})

const workerSeverity = computed<MetricTone>(() => (workerOnline.value ? 'good' : 'bad'))
const realtimeSeverity = computed<MetricTone>(() => (realtimeHealthy.value ? 'good' : 'warn'))
const storageSeverity = computed<MetricTone>(() => (storageApiReachable.value ? 'good' : 'bad'))

const successHint = computed(() => {
  if (successRate.value < 70) return '低于安全线'
  if (successRate.value < 90) return '需要盯紧'
  return '状态稳定'
})

const hexToRgba = (hex: string, alpha: number) => {
  const normalized = hex.replace('#', '')
  const full = normalized.length === 3
    ? normalized
        .split('')
        .map((char) => char + char)
        .join('')
    : normalized

  const value = Number.parseInt(full, 16)
  const r = (value >> 16) & 255
  const g = (value >> 8) & 255
  const b = value & 255

  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

const chartTheme = computed(() => ({
  textStrong: isDarkTheme.value ? '#eef4fb' : '#243447',
  textMuted: isDarkTheme.value ? '#9eb0c5' : '#6a7d92',
  axisLine: isDarkTheme.value ? 'rgba(190, 206, 224, 0.24)' : 'rgba(110, 130, 152, 0.2)',
  splitLine: isDarkTheme.value ? 'rgba(190, 206, 224, 0.1)' : 'rgba(110, 130, 152, 0.1)',
  area: hexToRgba(accentColor.value, isDarkTheme.value ? 0.18 : 0.14),
  accentSoft: hexToRgba(accentColor.value, isDarkTheme.value ? 0.26 : 0.18),
  paper: isDarkTheme.value ? '#101722' : '#f4f8fc',
  pie: ['#8aa0bb', accentColor.value, '#78b39d', '#d27070'],
}))

const dbRowsChartOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  grid: { left: 8, right: 14, top: 12, bottom: 10, outerBoundsMode: 'same' as const },
  xAxis: {
    type: 'value',
    axisLabel: { color: chartTheme.value.textMuted },
    axisLine: { lineStyle: { color: chartTheme.value.axisLine } },
    splitLine: { lineStyle: { color: chartTheme.value.splitLine } },
  },
  yAxis: {
    type: 'category',
    axisLabel: { color: chartTheme.value.textStrong },
    axisLine: { show: false },
    data: ['processing_tasks', 'model_assets', 'memory_poses', 'rag_docs', 'tasks'],
  },
  series: [
    {
      type: 'bar',
      barWidth: 14,
      itemStyle: { color: accentColor.value, borderRadius: [0, 10, 10, 0] },
      emphasis: { itemStyle: { color: '#f2a33c' } },
      data: [
        dbCounts.value.processing_tasks,
        dbCounts.value.model_assets,
        dbCounts.value.memory_poses,
        dbCounts.value.rag_docs,
        dbCounts.value.tasks,
      ],
    },
  ],
}))

const taskTrendOption = computed(() => {
  const labels: string[] = []
  const bucketMap: Record<string, number> = {}
  let rangeLabel = '最近 24 小时'

  const now = dayjs()

  if (taskTrendRange.value === '24h') {
    rangeLabel = '最近 24 小时'
    for (let i = 23; i >= 0; i -= 1) {
      const key = now.subtract(i, 'hour').startOf('hour').format('MM-DD HH:00')
      labels.push(key)
      bucketMap[key] = 0
    }
    taskRows.value.forEach((item) => {
      const key = dayjs(item.created_at).startOf('hour').format('MM-DD HH:00')
      if (key in bucketMap) bucketMap[key] = (bucketMap[key] ?? 0) + 1
    })
  } else if (taskTrendRange.value === '7d') {
    rangeLabel = '最近 7 天'
    for (let i = 6; i >= 0; i -= 1) {
      const key = now.subtract(i, 'day').startOf('day').format('MM-DD')
      labels.push(key)
      bucketMap[key] = 0
    }
    taskRows.value.forEach((item) => {
      const key = dayjs(item.created_at).startOf('day').format('MM-DD')
      if (key in bucketMap) bucketMap[key] = (bucketMap[key] ?? 0) + 1
    })
  } else if (taskTrendRange.value === '30d') {
    rangeLabel = '最近 30 天'
    for (let i = 29; i >= 0; i -= 1) {
      const key = now.subtract(i, 'day').startOf('day').format('MM-DD')
      labels.push(key)
      bucketMap[key] = 0
    }
    taskRows.value.forEach((item) => {
      const key = dayjs(item.created_at).startOf('day').format('MM-DD')
      if (key in bucketMap) bucketMap[key] = (bucketMap[key] ?? 0) + 1
    })
  } else {
    const sorted = [...taskRows.value].sort((a, b) => dayjs(a.created_at).valueOf() - dayjs(b.created_at).valueOf())
    if (!sorted.length) {
      const onlyKey = now.format('MM-DD')
      labels.push(onlyKey)
      bucketMap[onlyKey] = 0
    } else {
      const firstTask = sorted[0]
      const lastTask = sorted[sorted.length - 1]
      if (!firstTask || !lastTask) {
        const onlyKey = now.format('MM-DD')
        labels.push(onlyKey)
        bucketMap[onlyKey] = 0
      } else {
        const first = dayjs(firstTask.created_at)
        const last = dayjs(lastTask.created_at)
        const daysSpan = Math.max(last.diff(first, 'day'), 0)

        if (daysSpan <= 90) {
          rangeLabel = '全部（按天）'
          let cursor = first.startOf('day')
          const end = last.startOf('day')
          while (cursor.isBefore(end) || cursor.isSame(end)) {
            const key = cursor.format('MM-DD')
            labels.push(key)
            bucketMap[key] = 0
            cursor = cursor.add(1, 'day')
          }
          taskRows.value.forEach((item) => {
            const key = dayjs(item.created_at).startOf('day').format('MM-DD')
            if (key in bucketMap) bucketMap[key] = (bucketMap[key] ?? 0) + 1
          })
        } else {
          rangeLabel = '全部（按月）'
          let cursor = first.startOf('month')
          const end = last.startOf('month')
          while (cursor.isBefore(end) || cursor.isSame(end)) {
            const key = cursor.format('YYYY-MM')
            labels.push(key)
            bucketMap[key] = 0
            cursor = cursor.add(1, 'month')
          }
          taskRows.value.forEach((item) => {
            const key = dayjs(item.created_at).startOf('month').format('YYYY-MM')
            if (key in bucketMap) bucketMap[key] = (bucketMap[key] ?? 0) + 1
          })
        }
      }
    }
  }

  return {
    tooltip: { trigger: 'axis' },
    legend: {
      data: [rangeLabel],
      top: 0,
      textStyle: { color: chartTheme.value.textMuted, fontSize: 12 },
    },
    grid: { left: 12, right: 18, top: 34, bottom: 12, outerBoundsMode: 'same' as const },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: chartTheme.value.textMuted, hideOverlap: true },
      axisLine: { lineStyle: { color: chartTheme.value.axisLine } },
    },
    yAxis: {
      type: 'value',
      minInterval: 1,
      axisLabel: { color: chartTheme.value.textMuted },
      splitLine: { lineStyle: { color: chartTheme.value.splitLine } },
    },
    series: [
      {
        name: rangeLabel,
        type: 'line',
        smooth: 0.35,
        data: labels.map((key) => bucketMap[key]),
        symbolSize: 7,
        lineStyle: { width: 3, color: accentColor.value },
        itemStyle: { color: '#f2a33c', borderColor: accentColor.value, borderWidth: 2 },
        areaStyle: { color: chartTheme.value.area },
      },
    ],
  }
})

const statusPieOption = computed(() => ({
  tooltip: { trigger: 'item' },
  legend: {
    bottom: 0,
    itemGap: 18,
    textStyle: { color: chartTheme.value.textMuted, fontSize: 12 },
  },
  series: [
    {
      type: 'pie',
      radius: ['46%', '74%'],
      center: ['50%', '42%'],
      itemStyle: {
        borderRadius: 10,
        borderColor: chartTheme.value.paper,
        borderWidth: 2,
      },
      label: {
        show: true,
        formatter: '{b}\n{d}%',
        color: chartTheme.value.textStrong,
        fontSize: 12,
      },
      labelLine: { lineStyle: { color: chartTheme.value.axisLine } },
      data: [
        { value: pendingCount.value, name: '排队中', itemStyle: { color: chartTheme.value.pie[0] } },
        { value: processingCount.value, name: '处理中', itemStyle: { color: chartTheme.value.pie[1] } },
        { value: completedCount.value, name: '已完成', itemStyle: { color: chartTheme.value.pie[2] } },
        { value: failedCount.value, name: '失败', itemStyle: { color: chartTheme.value.pie[3] } },
      ],
    },
  ],
}))

const formatDisplayName = (task: ProcessingTask) => task.display_name || task.scene_id || task.id
const formatDateTime = (iso: string) => dayjs(iso).format('YYYY-MM-DD HH:mm:ss')

const getProgressByStatus = (status: TaskStatus) => {
  if (status === 'pending') return 15
  if (status === 'processing') return 65
  return 100
}

const getLatestLogMessage = (logs: unknown) => {
  if (!Array.isArray(logs) || logs.length === 0) return '无日志'
  const last = logs[logs.length - 1] as { msg?: string }
  return (last?.msg || '无日志').slice(0, 100)
}

const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  const mb = kb / 1024
  if (mb < 1024) return `${mb.toFixed(1)} MB`
  return `${(mb / 1024).toFixed(2)} GB`
}

const applyTheme = () => {
  document.documentElement.classList.toggle('theme-dark', isDarkTheme.value)
  document.documentElement.classList.toggle('theme-light', !isDarkTheme.value)
  document.documentElement.style.setProperty('--accent-color', accentColor.value)
}

const edgeFunctionNames = computed(() => {
  const raw = (import.meta.env.VITE_SUPABASE_EDGE_FUNCTIONS as string | undefined) ?? 'search-models,test-timeout'
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
})

const edgeStatusText = computed(() => {
  if (!edgeChecks.value.length) return '还没探测'
  if (edgeHealthyCount.value === edgeFunctionNames.value.length) return '全部在线'
  if (!edgeHealthyCount.value) return '全部失联'
  return '有函数掉线'
})

const edgeSeverity = computed<MetricTone>(() => {
  if (!edgeFunctionNames.value.length) return 'warn'
  if (edgeHealthyCount.value === edgeFunctionNames.value.length) return 'good'
  if (!edgeHealthyCount.value) return 'bad'
  return 'warn'
})

const overviewCards = computed(() => [
  {
    key: 'success',
    label: '成功率',
    value: `${successRate.value}%`,
    note: successHint.value,
    icon: 'lucide:gauge',
    tone: successSeverity.value,
  },
  {
    key: 'failed',
    label: '失败任务',
    value: `${failedCount.value}`,
    note: failedCount.value ? '先查失败列表' : '当前无失败',
    icon: 'lucide:octagon-alert',
    tone: failureSeverity.value,
  },
  {
    key: 'queue',
    label: '队列总数',
    value: `${queueCount.value}`,
    note: `排队 ${pendingCount.value} / 处理 ${processingCount.value}`,
    icon: 'lucide:list-todo',
    tone: queueSeverity.value,
  },
  {
    key: 'worker',
    label: 'Worker',
    value: workerOnline.value ? '在线' : '离线',
    note: taskFreshnessText.value,
    icon: 'lucide:bot',
    tone: workerSeverity.value,
  },
  {
    key: 'storage',
    label: 'Storage 状态',
    value: formatBytes(totalStorageBytes.value),
    note: `${storageStats.value.length} 个桶`,
    icon: 'lucide:database',
    tone: storageSeverity.value,
  },
  {
    key: 'edge',
    label: 'Edge',
    value: `${edgeHealthyCount.value}/${edgeFunctionNames.value.length}`,
    note: edgeStatusText.value,
    icon: 'lucide:plug-zap',
    tone: edgeSeverity.value,
  },
  {
    key: 'assets',
    label: '模型资产',
    value: `${modelAssetCount.value}`,
    note: `姿态 ${memoryPoseCount.value}`,
    icon: 'lucide:boxes',
    tone: 'neutral' as const,
  },
  {
    key: 'quality',
    label: '平均质量',
    value: `${avgQualityScore.value}`,
    note: `均耗时 ${avgProcessMinutes.value}`,
    icon: 'lucide:badge-check',
    tone: 'neutral' as const,
  },
])

const alertRows = computed(() => [
  {
    key: 'success',
    label: '任务成功率',
    value: `${successRate.value}%`,
    note: successHint.value,
    icon: 'lucide:gauge',
    tone: successSeverity.value,
  },
  {
    key: 'failed',
    label: '失败任务',
    value: `${failedCount.value}`,
    note: failedCount.value ? '先查失败列表' : '当前无失败',
    icon: 'lucide:triangle-alert',
    tone: failureSeverity.value,
  },
  {
    key: 'realtime',
    label: '实时链路',
    value: realtimeStatusText.value,
    note: realtimeHealthy.value ? '订阅都在线' : '有频道重连',
    icon: 'lucide:radio-tower',
    tone: realtimeSeverity.value,
  },
  {
    key: 'worker',
    label: 'Worker',
    value: workerOnline.value ? '在线' : '离线',
    note: taskFreshnessText.value,
    icon: 'lucide:bot',
    tone: workerSeverity.value,
  },
  {
    key: 'edge',
    label: 'Edge',
    value: `${edgeHealthyCount.value}/${edgeFunctionNames.value.length}`,
    note: edgeStatusText.value,
    icon: 'lucide:plug-zap',
    tone: edgeSeverity.value,
  },
  {
    key: 'storage',
    label: 'Storage 状态',
    value: storageApiReachable.value ? '可读' : '受限',
    note: storageModeText.value,
    icon: 'lucide:hard-drive',
    tone: storageSeverity.value,
  },
])

const activeAlertCount = computed(() =>
  alertRows.value.filter((item) => item.tone === 'warn' || item.tone === 'bad').length,
)

const checkEdgeFunction = async (name: string): Promise<EdgeFunctionCheck> => {
  const baseUrl = (import.meta.env.VITE_SUPABASE_URL as string | undefined) ?? ''
  const anonKey = (import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined) ?? ''
  const start = performance.now()
  const now = dayjs().format('YYYY-MM-DD HH:mm:ss')

  if (!baseUrl) {
    return {
      name,
      status: 'error',
      httpStatus: null,
      latencyMs: null,
      lastCheckedAt: now,
      message: '缺少 VITE_SUPABASE_URL',
    }
  }

  const url = `${baseUrl.replace(/\/$/, '')}/functions/v1/${name}`
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 5000)

  try {
    const res = await fetch(url, {
      method: 'OPTIONS',
      headers: {
        apikey: anonKey,
        Authorization: anonKey ? `Bearer ${anonKey}` : '',
      },
      signal: controller.signal,
    })
    window.clearTimeout(timer)
    const latency = Math.round(performance.now() - start)

    if (res.status === 404) {
      return {
        name,
        status: 'missing',
        httpStatus: res.status,
        latencyMs: latency,
        lastCheckedAt: now,
        message: '函数未部署或名称不匹配',
      }
    }

    return {
      name,
      status: 'ok',
      httpStatus: res.status,
      latencyMs: latency,
      lastCheckedAt: now,
      message: '可达',
    }
  } catch (error) {
    window.clearTimeout(timer)
    return {
      name,
      status: 'error',
      httpStatus: null,
      latencyMs: Math.round(performance.now() - start),
      lastCheckedAt: now,
      message: error instanceof Error ? error.message : '请求失败',
    }
  }
}

const refreshEdgeChecks = async () => {
  edgeLoading.value = true
  const checks = await Promise.all(edgeFunctionNames.value.map((name) => checkEdgeFunction(name)))
  edgeChecks.value = checks
  edgeLoading.value = false
}

const scanBucket = async (bucketId: string, isPublic: boolean): Promise<BucketProbeResult> => {
  const objects: StorageListObject[] = []
  const dirsToScan: string[] = ['']
  const scannedDirs = new Set<string>()
  const limit = 200
  const maxScan = 4000
  let ok = true
  let scannedObjects = 0

  while (dirsToScan.length > 0 && scannedObjects < maxScan) {
    const dir = dirsToScan.shift() ?? ''
    if (scannedDirs.has(dir)) continue
    scannedDirs.add(dir)

    let offset = 0
    while (offset < maxScan && scannedObjects < maxScan) {
      const listRes = await supabase.storage.from(bucketId).list(dir, {
        limit,
        offset,
        sortBy: { column: 'updated_at', order: 'desc' },
      })

      if (listRes.error) {
        ok = false
        break
      }

      const rows = (listRes.data ?? []) as unknown as StorageListObject[]
      for (const row of rows) {
        const fullName = dir ? `${dir}/${row.name}` : row.name
        const isFile = typeof row.metadata?.size === 'number'
        if (isFile) {
          objects.push({ ...row, name: fullName })
          scannedObjects += 1
        } else {
          dirsToScan.push(fullName)
        }
      }

      if (rows.length < limit) break
      offset += limit
    }
    if (!ok) break
  }

  const totalBytes = objects.reduce((sum, item) => {
    return sum + (typeof item.metadata?.size === 'number' ? item.metadata.size : 0)
  }, 0)

  const latest =
    objects
      .map((obj) => obj.updated_at)
      .filter((v): v is string => Boolean(v))
      .sort((a, b) => dayjs(b).valueOf() - dayjs(a).valueOf())[0] || null

  return {
    ok,
    stat: {
      id: bucketId,
      public: isPublic,
      objectCount: objects.length,
      totalBytes,
      latestUpdatedAt: latest,
    },
  }
}

const fetchStorageStats = async () => {
  storageLoading.value = true
  storageError.value = ''
  storageProbeMode.value = 'none'

  const bucketRes = await supabase.storage.listBuckets()
  if (!bucketRes.error) {
    storageProbeMode.value = 'bucket_list'
    storageApiReachable.value = true

    const stats = await Promise.all(
      (bucketRes.data ?? []).map(async (bucket) => {
        const result = await scanBucket(bucket.id, Boolean(bucket.public))
        return result.stat
      }),
    )

    storageStats.value = stats.sort((a, b) => b.totalBytes - a.totalBytes)
    storageLoading.value = false
    return
  }

  const knownBuckets = (import.meta.env.VITE_STORAGE_BUCKETS as string | undefined)
    ?.split(',')
    .map((item) => item.trim())
    .filter(Boolean) ?? ['braindance-assets']

  const probed = await Promise.all(knownBuckets.map((bucket) => scanBucket(bucket, true)))
  const readable = probed.filter((item) => item.ok).map((item) => item.stat)

  if (readable.length > 0) {
    storageProbeMode.value = 'known_buckets'
    storageApiReachable.value = true
    storageError.value = `无法列出桶（${bucketRes.error.message}），已切换为已知桶探测模式。`
    storageStats.value = readable.sort((a, b) => b.totalBytes - a.totalBytes)
  } else {
    storageApiReachable.value = false
    storageError.value = `无法列出桶且已知桶探测失败。请检查 Storage policy（buckets/objects 的 select 权限）或改用管理员登录。`
    storageStats.value = []
  }

  storageLoading.value = false
}

const refreshDashboard = async () => {
  if (!loading.value) refreshing.value = true
  errorMessage.value = ''

  const now = dayjs()
  const since24h = now.subtract(24, 'hour').toISOString()
  const since7d = now.subtract(7, 'day').toISOString()

  const [tasksRes, assetCountRes, poseCountRes, ragCountRes, taskTableRes, task24hRes, asset7dRes] = await Promise.all([
    supabase
      .from('processing_tasks')
      .select('id, display_name, scene_id, user_id, status, task_type, quality_score, created_at, updated_at, logs')
      .order('updated_at', { ascending: false })
      .limit(500),
    supabase.from('model_assets').select('*', { count: 'exact', head: true }),
    supabase.from('memory_poses').select('*', { count: 'exact', head: true }),
    supabase.from('rag_docs').select('*', { count: 'exact', head: true }),
    supabase.from('tasks').select('*', { count: 'exact', head: true }),
    supabase
      .from('processing_tasks')
      .select('status, user_id, created_at', { count: 'exact' })
      .gte('created_at', since24h)
      .limit(1000),
    supabase.from('model_assets').select('created_at', { count: 'exact', head: true }).gte('created_at', since7d),
  ])

  if (tasksRes.error || assetCountRes.error || poseCountRes.error) {
    errorMessage.value = tasksRes.error?.message || assetCountRes.error?.message || poseCountRes.error?.message || '数据读取失败'
  } else {
    const tasks = (tasksRes.data ?? []) as ProcessingTask[]
    taskRows.value = tasks
    modelAssetCount.value = assetCountRes.count ?? 0
    memoryPoseCount.value = poseCountRes.count ?? 0

    const tasks24hRows = (task24hRes.data ?? []) as Array<{ status: string; user_id: string }>
    timeBasedStats.value = {
      tasks24h: task24hRes.count ?? 0,
      failed24h: tasks24hRows.filter((item) => item.status === 'failed').length,
      completed24h: tasks24hRows.filter((item) => item.status === 'completed').length,
      activeUsers7d: new Set(
        tasks.filter((item) => dayjs(item.created_at).isAfter(dayjs(since7d))).map((item) => item.user_id),
      ).size,
      assets7d: asset7dRes.count ?? 0,
    }

    dbCounts.value = {
      processing_tasks: tasksRes.data?.length ?? 0,
      model_assets: assetCountRes.count ?? 0,
      memory_poses: poseCountRes.count ?? 0,
      rag_docs: ragCountRes.error ? 0 : (ragCountRes.count ?? 0),
      tasks: taskTableRes.error ? 0 : (taskTableRes.count ?? 0),
    }

    lastUpdated.value = dayjs().format('YYYY-MM-DD HH:mm:ss')
  }

  await Promise.all([fetchStorageStats(), refreshEdgeChecks()])

  loading.value = false
  refreshing.value = false
}

let refreshTimer: number | undefined
let pollTimer: number | undefined
const channels: RealtimeChannel[] = []

const scheduleRefresh = () => {
  if (refreshTimer) window.clearTimeout(refreshTimer)
  refreshTimer = window.setTimeout(() => {
    void refreshDashboard()
  }, 350)
}

const restartPolling = () => {
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = undefined
  }

  if (autoRefresh.value) {
    pollTimer = window.setInterval(() => {
      void refreshDashboard()
    }, refreshSeconds.value * 1000)
  }
}

const bindChannel = (tableName: string, channelName: string) => {
  const channel = supabase
    .channel(channelName)
    .on('postgres_changes', { event: '*', schema: 'public', table: tableName }, scheduleRefresh)
    .subscribe((status) => {
      channelState.value[tableName] = status
    })

  channels.push(channel)
}

watch([autoRefresh, refreshSeconds], () => {
  restartPolling()
})

onMounted(async () => {
  const savedDark = localStorage.getItem(THEME_STORAGE_KEY)
  const savedAccent = localStorage.getItem(ACCENT_STORAGE_KEY)
  if (savedDark === '1') {
    isDarkTheme.value = true
  } else if (savedDark === '0') {
    isDarkTheme.value = false
  }
  if (savedAccent) {
    accentColor.value = savedAccent
  }
  applyTheme()

  await refreshDashboard()

  bindChannel('processing_tasks', 'dashboard-processing-tasks')
  bindChannel('model_assets', 'dashboard-model-assets')
  bindChannel('memory_poses', 'dashboard-memory-poses')

  restartPolling()
})

watch([isDarkTheme, accentColor], () => {
  localStorage.setItem(THEME_STORAGE_KEY, isDarkTheme.value ? '1' : '0')
  localStorage.setItem(ACCENT_STORAGE_KEY, accentColor.value)
  applyTheme()
})

onUnmounted(() => {
  if (refreshTimer) window.clearTimeout(refreshTimer)
  if (pollTimer) window.clearInterval(pollTimer)
  channels.forEach((channel) => {
    void supabase.removeChannel(channel)
  })
})
</script>

<template>
  <div class="dashboard-page">
    <section class="shell-grid">
      <aside class="phone-shell">
        <div class="phone-shell__glow"></div>
        <div class="phone-shell__frame">
          <div class="phone-shell__head">
            <div>
              <p class="eyebrow">BrainDance</p>
              <h1>Dashboard</h1>
            </div>
            <div class="status-dot" :class="`tone-${realtimeSeverity}`">
              <Icon icon="lucide:radio-tower" />
              <span>{{ realtimeStatusText }}</span>
            </div>
          </div>

          <div class="phone-hero">
            <div class="phone-hero__badge">实时总览</div>
            <strong>{{ successRate }}%</strong>
            <span>任务成功率</span>
            <p>{{ successHint }}，最近 24 小时完成 {{ timeBasedStats.completed24h }} 条。</p>
          </div>

          <div class="phone-stats">
            <article class="phone-stat-card">
              <span>排队</span>
              <strong>{{ pendingCount }}</strong>
            </article>
            <article class="phone-stat-card">
              <span>处理中</span>
              <strong>{{ processingCount }}</strong>
            </article>
            <article class="phone-stat-card">
              <span>失败</span>
              <strong>{{ failedCount }}</strong>
            </article>
            <article class="phone-stat-card">
              <span>资产</span>
              <strong>{{ modelAssetCount }}</strong>
            </article>
          </div>

          <div class="phone-panel">
            <div class="phone-panel__title">
              <span>Live status</span>
              <strong>{{ taskFreshnessText }}</strong>
            </div>
            <div class="phone-list">
              <div class="phone-list__item">
                <Icon icon="lucide:clock-3" />
                <div>
                  <span>最后采样</span>
                  <strong>{{ lastUpdated ?? '还没采样' }}</strong>
                </div>
              </div>
              <div class="phone-list__item">
                <Icon icon="lucide:refresh-cw" />
                <div>
                  <span>刷新节奏</span>
                  <strong>{{ refreshModeText }}</strong>
                </div>
              </div>
              <div class="phone-list__item">
                <Icon icon="lucide:database" />
                <div>
                  <span>Storage</span>
                  <strong>{{ storageApiReachable ? '可读' : '受限' }}</strong>
                </div>
              </div>
            </div>
          </div>

          <div class="bottom-dock">
            <div class="bottom-dock__item bottom-dock__item--active">
              <Icon icon="lucide:layout-dashboard" />
              <span>概览</span>
            </div>
            <div class="bottom-dock__item">
              <Icon icon="lucide:activity" />
              <span>趋势</span>
            </div>
            <div class="bottom-dock__item">
              <Icon icon="lucide:database-zap" />
              <span>资源</span>
            </div>
            <div class="bottom-dock__item">
              <Icon icon="lucide:settings-2" />
              <span>设置</span>
            </div>
          </div>
        </div>
      </aside>

      <main class="content-stage">
        <section class="hero-card glass-card">
          <div class="hero-card__copy">
            <p class="eyebrow">BrainDance Operations</p>
            <h2>统一查看任务、资源与服务状态</h2>
            <p class="hero-card__text">
              面向运营与排障场景，集中展示任务成功率、队列压力、存储可用性和实时连接状态。
            </p>

            <div class="hero-card__actions">
              <el-button :loading="refreshing" type="primary" @click="refreshDashboard">
                <Icon icon="lucide:refresh-cw" />
                <span>立即刷新</span>
              </el-button>

              <div class="theme-pill">
                <Icon :icon="isDarkTheme ? 'lucide:moon-star' : 'lucide:sun-medium'" />
                <el-switch v-model="isDarkTheme" inline-prompt active-text="夜间" inactive-text="日间" />
              </div>

              <div class="theme-pill">
                <Icon icon="lucide:paintbrush-2" />
                <el-color-picker
                  v-model="accentColor"
                  :predefine="['#71839a', '#5f86c2', '#86a8a1', '#8d9bc4', '#a0afc7']"
                />
              </div>
            </div>
          </div>

          <div class="hero-card__summary">
            <article class="summary-pill" :class="`tone-${successSeverity}`">
              <span>成功率</span>
              <strong>{{ successRate }}%</strong>
            </article>
            <article class="summary-pill" :class="`tone-${queueSeverity}`">
              <span>队列</span>
              <strong>{{ queueCount }}</strong>
            </article>
            <article class="summary-pill" :class="`tone-${workerSeverity}`">
              <span>Worker</span>
              <strong>{{ workerOnline ? '在线' : '离线' }}</strong>
            </article>
          </div>
        </section>

        <section class="overview-grid">
          <article
            v-for="item in overviewCards"
            :key="item.key"
            class="overview-card glass-card"
            :class="`tone-${item.tone}`"
          >
            <div class="overview-card-top">
              <div class="icon-chip">
                <Icon :icon="item.icon" />
              </div>
              <span class="overview-label">{{ item.label }}</span>
            </div>
            <div class="overview-value">{{ item.value }}</div>
            <p class="overview-note">{{ item.note }}</p>
          </article>
        </section>

        <el-alert v-if="errorMessage" :title="errorMessage" type="error" show-icon class="mb-16" />
        <el-alert v-if="storageError" :title="`Storage: ${storageError}`" type="warning" show-icon class="mb-16" />
        <el-skeleton v-if="loading" :rows="7" animated />

        <template v-else>
          <section class="filters-row glass-card">
            <div class="section-heading-main">
              <div class="section-icon-shell">
                <Icon icon="lucide:sliders-horizontal" />
              </div>
              <div>
                <span class="section-kicker">Filters</span>
                <h3 class="filters-title">筛选与刷新</h3>
              </div>
            </div>

            <div class="filters-controls">
              <el-select v-model="selectedStatus" class="ctrl" placeholder="状态过滤">
                <el-option label="全部状态" value="all" />
                <el-option label="排队中" value="pending" />
                <el-option label="处理中" value="processing" />
                <el-option label="已完成" value="completed" />
                <el-option label="失败" value="failed" />
              </el-select>

              <el-select v-model="selectedTaskType" class="ctrl" placeholder="任务类型过滤">
                <el-option label="全部类型" value="all" />
                <el-option v-for="item in taskTypeOptions" :key="item" :label="item" :value="item" />
              </el-select>

              <el-input v-model="searchKeyword" class="ctrl search" clearable placeholder="搜索任务 / 场景 / 用户" />

              <div class="inline-ops">
                <el-switch v-model="autoRefresh" inline-prompt active-text="自动" inactive-text="手动" />
                <el-select v-model="refreshSeconds" class="interval" :disabled="!autoRefresh">
                  <el-option :value="15" label="15s" />
                  <el-option :value="30" label="30s" />
                  <el-option :value="60" label="60s" />
                  <el-option :value="120" label="120s" />
                </el-select>
              </div>

              <div class="filter-count">{{ filteredTasks.length }} 条结果</div>
            </div>
          </section>

          <section class="panel-grid panel-grid--charts">
            <el-card shadow="never" class="chart-card glass-card chart-card--wide">
              <template #header>
                <div class="card-header-row">
                  <div>
                    <div class="card-header">任务趋势</div>
                    <div class="header-meta">对齐 app 的柔和蓝灰节奏感。</div>
                  </div>
                  <el-radio-group v-model="taskTrendRange" size="small">
                    <el-radio-button label="24h" value="24h">24h</el-radio-button>
                    <el-radio-button label="7d" value="7d">7d</el-radio-button>
                    <el-radio-button label="30d" value="30d">30d</el-radio-button>
                    <el-radio-button label="all" value="all">全部</el-radio-button>
                  </el-radio-group>
                </div>
              </template>
              <v-chart class="chart" :option="taskTrendOption" autoresize />
            </el-card>

            <el-card shadow="never" class="chart-card glass-card chart-card--compact">
              <template #header>
                <div>
                  <div class="card-header">状态占比</div>
                  <div class="header-meta">快速读出失败与完成比例。</div>
                </div>
              </template>
              <v-chart class="chart pie" :option="statusPieOption" autoresize />
            </el-card>
          </section>

          <section class="panel-grid panel-grid--operations">
            <el-card shadow="never" class="table-card glass-card">
              <template #header>
                <div>
                  <div class="card-header">任务队列</div>
                  <div class="header-meta">显示筛选后的前 20 条任务。</div>
                </div>
              </template>
              <el-table :data="taskQueue" stripe height="460" empty-text="没找到任务">
                <el-table-column label="任务名" min-width="220">
                  <template #default="scope">
                    <div class="task-name">{{ formatDisplayName(scope.row) }}</div>
                    <div class="task-sub">{{ scope.row.task_type || 'video_3dgs' }} / {{ scope.row.scene_id }}</div>
                  </template>
                </el-table-column>
                <el-table-column label="状态" width="110" align="center">
                  <template #default="scope">
                    <el-tag :type="statusMap[scope.row.status]?.type || 'info'">
                      {{ statusMap[scope.row.status]?.label || scope.row.status }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="进度" min-width="140">
                  <template #default="scope">
                    <el-progress
                      :percentage="getProgressByStatus(scope.row.status)"
                      :status="scope.row.status === 'failed' ? 'exception' : undefined"
                    />
                  </template>
                </el-table-column>
                <el-table-column label="质量" width="85" align="center">
                  <template #default="scope">
                    {{ typeof scope.row.quality_score === 'number' ? scope.row.quality_score : '-' }}
                  </template>
                </el-table-column>
                <el-table-column label="更新时间" min-width="165">
                  <template #default="scope">
                    {{ formatDateTime(scope.row.updated_at) }}
                  </template>
                </el-table-column>
              </el-table>
            </el-card>

            <el-card shadow="never" class="fail-card glass-card">
              <template #header>
                <div>
                  <div class="card-header">异常摘要</div>
                  <div class="header-meta">{{ activeAlertCount }} 项需要关注。</div>
                </div>
              </template>

              <div class="alerts-list alerts-list--soft">
                <article
                  v-for="item in alertRows"
                  :key="item.key"
                  class="alert-item"
                  :class="`tone-${item.tone}`"
                >
                  <div class="alert-item-top">
                    <div class="icon-chip icon-chip--small">
                      <Icon :icon="item.icon" />
                    </div>
                    <div>
                      <span class="alert-label">{{ item.label }}</span>
                      <strong class="alert-value">{{ item.value }}</strong>
                    </div>
                  </div>
                  <p class="alert-note">{{ item.note }}</p>
                </article>
              </div>

              <el-divider />

              <el-empty v-if="!failedTasks.length" description="暂无失败任务" />
              <el-timeline v-else>
                <el-timeline-item
                  v-for="item in failedTasks"
                  :key="item.id"
                  type="danger"
                  :timestamp="formatDateTime(item.updated_at)"
                >
                  <div class="fail-title">{{ formatDisplayName(item) }}</div>
                  <div class="fail-sub">{{ item.scene_id }} / {{ item.user_id }}</div>
                  <div class="fail-log">{{ getLatestLogMessage(item.logs) }}</div>
                </el-timeline-item>
              </el-timeline>
            </el-card>
          </section>

          <section class="panel-grid panel-grid--resources">
            <el-card shadow="never" class="storage-card glass-card">
              <template #header>
                <div>
                  <div class="card-header">Storage 状态</div>
                  <div class="header-meta">前端扫描的桶与体积估算。</div>
                </div>
              </template>
              <el-table :data="storageStats" stripe :loading="storageLoading" height="300" empty-text="桶还读不到">
                <el-table-column label="Bucket" min-width="180" prop="id" />
                <el-table-column label="可见性" width="90" align="center">
                  <template #default="scope">
                    <el-tag :type="scope.row.public ? 'success' : 'info'">{{ scope.row.public ? 'Public' : 'Private' }}</el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="对象数" width="95" align="right" prop="objectCount" />
                <el-table-column label="估算体积" min-width="120" align="right">
                  <template #default="scope">{{ formatBytes(scope.row.totalBytes) }}</template>
                </el-table-column>
                <el-table-column label="最近更新" min-width="170">
                  <template #default="scope">{{ scope.row.latestUpdatedAt ? formatDateTime(scope.row.latestUpdatedAt) : '-' }}</template>
                </el-table-column>
              </el-table>
            </el-card>

            <el-card shadow="never" class="db-card glass-card">
              <template #header>
                <div>
                  <div class="card-header">数据库概览</div>
                  <div class="header-meta">短周期活跃度与总量。</div>
                </div>
              </template>
              <div class="db-metrics">
                <div class="metric-item">
                  <div class="metric-title">24h 任务</div>
                  <div class="metric-value">{{ timeBasedStats.tasks24h }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">24h 失败</div>
                  <div class="metric-value bad">{{ timeBasedStats.failed24h }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">24h 完成</div>
                  <div class="metric-value ok">{{ timeBasedStats.completed24h }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">7d 活跃</div>
                  <div class="metric-value">{{ timeBasedStats.activeUsers7d }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">7d 资产</div>
                  <div class="metric-value">{{ timeBasedStats.assets7d }}</div>
                </div>
              </div>
              <v-chart class="db-chart" :option="dbRowsChartOption" autoresize />
            </el-card>
          </section>

          <section class="panel-grid panel-grid--edge">
            <el-card shadow="never" class="edge-card glass-card">
              <template #header>
                <div class="card-header-row">
                  <div>
                    <div class="card-header">Edge Functions</div>
                    <div class="header-meta">只测网关可达和延迟。</div>
                  </div>
                  <el-button size="small" :loading="edgeLoading" @click="refreshEdgeChecks">重新探测</el-button>
                </div>
              </template>
              <el-table :data="edgeChecks" stripe :loading="edgeLoading" height="290" empty-text="还没配函数名">
                <el-table-column label="函数名" min-width="150" prop="name" />
                <el-table-column label="状态" width="120" align="center">
                  <template #default="scope">
                    <el-tag :type="scope.row.status === 'ok' ? 'success' : scope.row.status === 'missing' ? 'warning' : 'danger'">
                      {{ scope.row.status === 'ok' ? '可达' : scope.row.status === 'missing' ? '未部署' : '异常' }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="HTTP" width="90" align="center">
                  <template #default="scope">{{ scope.row.httpStatus ?? '-' }}</template>
                </el-table-column>
                <el-table-column label="延迟(ms)" width="100" align="right">
                  <template #default="scope">{{ scope.row.latencyMs ?? '-' }}</template>
                </el-table-column>
                <el-table-column label="最近检查" min-width="160" prop="lastCheckedAt" />
                <el-table-column label="说明" min-width="220" prop="message" />
              </el-table>
            </el-card>
          </section>
        </template>
      </main>
    </section>
  </div>
</template>
