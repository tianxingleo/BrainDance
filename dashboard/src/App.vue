<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import type { RealtimeChannel } from '@supabase/supabase-js'
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
const isDarkTheme = ref(false)
const accentColor = ref('#18b2a6')

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

const dbRowsChartOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  grid: { left: 8, right: 16, top: 18, bottom: 10, outerBoundsMode: 'same' as const },
  xAxis: {
    type: 'value',
    axisLabel: { color: '#64748b' },
    splitLine: { lineStyle: { color: '#e2e8f0' } },
  },
  yAxis: {
    type: 'category',
    axisLabel: { color: '#334155' },
    data: ['processing_tasks', 'model_assets', 'memory_poses', 'rag_docs', 'tasks'],
  },
  series: [
    {
      type: 'bar',
      barWidth: 16,
      itemStyle: { color: '#0ea5e9', borderRadius: [0, 6, 6, 0] },
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
    legend: { data: [rangeLabel], textStyle: { color: '#334155' } },
    grid: { left: 12, right: 24, top: 36, bottom: 12, outerBoundsMode: 'same' as const },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: '#64748b', hideOverlap: true },
      axisLine: { lineStyle: { color: '#cbd5e1' } },
    },
    yAxis: {
      type: 'value',
      minInterval: 1,
      axisLabel: { color: '#64748b' },
      splitLine: { lineStyle: { color: '#e2e8f0' } },
    },
    series: [
      {
        name: rangeLabel,
        type: 'line',
        smooth: 0.25,
        data: labels.map((key) => bucketMap[key]),
        symbolSize: 8,
        lineStyle: { width: 3, color: '#0f766e' },
        itemStyle: { color: '#14b8a6' },
        areaStyle: { color: 'rgba(20, 184, 166, 0.18)' },
      },
    ],
  }
})

const statusPieOption = computed(() => ({
  tooltip: { trigger: 'item' },
  legend: { bottom: 0, textStyle: { color: '#334155' } },
  series: [
    {
      type: 'pie',
      radius: ['48%', '72%'],
      center: ['50%', '45%'],
      itemStyle: { borderRadius: 8, borderColor: '#fff', borderWidth: 2 },
      label: { show: true, formatter: '{b}: {d}%', color: '#334155' },
      data: [
        { value: pendingCount.value, name: '排队中', itemStyle: { color: '#f59e0b' } },
        { value: processingCount.value, name: '处理中', itemStyle: { color: '#0ea5e9' } },
        { value: completedCount.value, name: '已完成', itemStyle: { color: '#10b981' } },
        { value: failedCount.value, name: '失败', itemStyle: { color: '#ef4444' } },
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
  document.documentElement.style.setProperty('--accent-color', accentColor.value)
}

const edgeFunctionNames = computed(() => {
  const raw = (import.meta.env.VITE_SUPABASE_EDGE_FUNCTIONS as string | undefined) ?? 'search-models,test-timeout'
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
})

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
    <header class="hero">
      <div>
        <p class="eyebrow">BrainDance Monitor</p>
        <h1>系统状态可视化看板</h1>
        <p class="hero-subtitle">新增 Supabase Storage 与数据库分析模块，实时展示可读取的接口与数据规模。</p>
        <div class="hero-pills">
          <span class="hero-pill">任务总量 {{ dbCounts.processing_tasks }}</span>
          <span class="hero-pill">24h 新任务 {{ timeBasedStats.tasks24h }}</span>
          <span class="hero-pill">资产 {{ dbCounts.model_assets }}</span>
          <span class="hero-pill">Storage {{ formatBytes(totalStorageBytes) }}</span>
        </div>
      </div>
      <div class="hero-actions">
        <el-button :loading="refreshing" type="primary" @click="refreshDashboard">手动刷新</el-button>
        <span class="refresh-time">最近刷新：{{ lastUpdated ?? '尚未刷新' }}</span>
        <div class="theme-controls">
          <el-switch v-model="isDarkTheme" inline-prompt active-text="夜间" inactive-text="白天" />
          <el-color-picker
            v-model="accentColor"
            :predefine="['#18b2a6', '#2b7fff', '#ff7a59', '#f59e0b', '#7c5cff']"
          />
        </div>
      </div>
    </header>

    <el-alert v-if="errorMessage" :title="errorMessage" type="error" show-icon class="mb-16" />
    <el-alert v-if="storageError" :title="`Storage API: ${storageError}`" type="warning" show-icon class="mb-16" />

    <el-skeleton v-if="loading" :rows="7" animated />

    <template v-else>
      <section class="stat-grid">
        <el-card class="stat-card" shadow="hover">
          <div class="kpi-title">Worker 心跳</div>
          <div class="kpi-value">{{ workerOnline ? '在线' : '离线' }}</div>
          <p class="stat-meta" :class="workerOnline ? 'ok' : 'bad'">{{ taskFreshnessText }}</p>
        </el-card>

        <el-card class="stat-card" shadow="hover">
          <div class="kpi-title">Realtime 状态</div>
          <div class="kpi-value">{{ realtimeStatusText }}</div>
          <p class="stat-meta" :class="realtimeHealthy ? 'ok' : 'bad'">
            tasks={{ channelState.processing_tasks }} | assets={{ channelState.model_assets }}
          </p>
        </el-card>

        <el-card class="stat-card" shadow="hover">
          <div class="kpi-title">Storage 接口</div>
          <div class="kpi-value">{{ storageApiReachable ? '可访问' : '不可访问' }}</div>
          <p class="stat-meta">
            总容量 {{ formatBytes(totalStorageBytes) }} / 模式
            {{ storageProbeMode === 'bucket_list' ? '列桶' : storageProbeMode === 'known_buckets' ? '已知桶探测' : '不可用' }}
          </p>
        </el-card>

        <el-card class="stat-card" shadow="hover">
          <el-statistic title="任务成功率" :value="successRate" suffix="%" />
          <p class="stat-meta">平均质量 {{ avgQualityScore }} / 平均耗时 {{ avgProcessMinutes }}</p>
        </el-card>
      </section>

      <section class="filters-row">
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

        <el-input v-model="searchKeyword" class="ctrl search" clearable placeholder="搜索任务名 / scene_id / user_id" />

        <div class="inline-ops">
          <el-switch v-model="autoRefresh" inline-prompt active-text="自动刷新" inactive-text="手动" />
          <el-select v-model="refreshSeconds" class="interval" :disabled="!autoRefresh">
            <el-option :value="15" label="15s" />
            <el-option :value="30" label="30s" />
            <el-option :value="60" label="60s" />
            <el-option :value="120" label="120s" />
          </el-select>
        </div>
      </section>

      <div class="section-title">任务运行态势</div>
      <section class="panel-grid">
        <el-card shadow="never" class="chart-card">
          <template #header>
            <div class="card-header-row">
              <div class="card-header">任务趋势</div>
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

        <el-card shadow="never" class="chart-card">
          <template #header>
            <div class="card-header">任务状态分布</div>
          </template>
          <v-chart class="chart pie" :option="statusPieOption" autoresize />
        </el-card>
      </section>

      <div class="section-title">故障与队列详情</div>
      <section class="panel-grid second">
        <el-card shadow="never" class="table-card">
          <template #header>
            <div class="card-header">任务队列（{{ filteredTasks.length }}）</div>
          </template>
          <el-table :data="taskQueue" stripe height="460" empty-text="没有匹配任务">
            <el-table-column label="任务名" min-width="200">
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

        <el-card shadow="never" class="fail-card">
          <template #header>
            <div class="card-header">最近失败任务摘要</div>
          </template>
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

      <div class="section-title">Supabase 资源分析</div>
      <section class="panel-grid third">
        <el-card shadow="never" class="storage-card">
          <template #header>
            <div class="card-header">Supabase Storage 状态分析</div>
          </template>
          <el-table :data="storageStats" stripe :loading="storageLoading" height="300" empty-text="无可读桶（请检查 Storage policy）">
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
          <p class="hint">说明：对象统计采用前端分页扫描（最多每桶 2000 条），用于运维观察，不等同账单精确值。</p>
        </el-card>

        <el-card shadow="never" class="db-card">
          <template #header>
            <div class="card-header">数据库分析（可读表）</div>
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
              <div class="metric-title">7d 活跃用户</div>
              <div class="metric-value">{{ timeBasedStats.activeUsers7d }}</div>
            </div>
            <div class="metric-item">
              <div class="metric-title">7d 新增资产</div>
              <div class="metric-value">{{ timeBasedStats.assets7d }}</div>
            </div>
          </div>
          <v-chart class="db-chart" :option="dbRowsChartOption" autoresize />
        </el-card>
      </section>

      <div class="section-title">Supabase Edge Functions</div>
      <section class="panel-grid fourth">
        <el-card shadow="never" class="edge-card">
          <template #header>
            <div class="card-header-row">
              <div class="card-header">Deno Edge Functions 状态</div>
              <el-button size="small" :loading="edgeLoading" @click="refreshEdgeChecks">刷新探测</el-button>
            </div>
          </template>
          <el-table :data="edgeChecks" stripe :loading="edgeLoading" height="290" empty-text="未配置函数名">
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
          <p class="hint">默认探测方法为 OPTIONS 请求，仅判断函数网关可达性；函数名来自 `VITE_SUPABASE_EDGE_FUNCTIONS`。</p>
        </el-card>
      </section>
    </template>
  </div>
</template>
