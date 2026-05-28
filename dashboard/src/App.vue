<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import type { RealtimeChannel } from '@supabase/supabase-js'
import { Icon } from '@iconify/vue'
import dayjs from 'dayjs'
import { ElMessage } from 'element-plus'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, PieChart, BarChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import TaskLogDrawer from './components/TaskLogDrawer.vue'
import { supabase } from './lib/supabase'
import { buildTaskInsight, type TaskInsight, type TaskStatus } from './lib/task-insights'

use([CanvasRenderer, LineChart, PieChart, BarChart, GridComponent, TooltipComponent, LegendComponent])

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

interface UserActivitySummary {
  userId: string
  displayName: string
  taskCount: number
  assetCount: number
  task24h: number
  task7d: number
  asset7d: number
  lastSeenAt: string | null
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

interface WorkerNode {
  worker_id: string
  hostname: string | null
  pid: number | null
  status: string
  current_task_id: string | null
  current_scene_id: string | null
  desired_state: 'run' | 'pause' | 'interrupt' | string
  control_note: string | null
  last_heartbeat: string
  started_at: string | null
  stopped_at: string | null
  metadata?: {
    online_timeout_seconds?: number
    stop_reason?: string | null
    desired_state_seen?: string | null
  } | null
}

interface UserActivityAggregateRow {
  user_id: string
  total_tasks: number
  tasks_24h: number
  tasks_7d: number
  total_assets: number
  assets_7d: number
  last_active: string | null
}

const loading = ref(true)
const refreshing = ref(false)
const storageLoading = ref(false)
const errorMessage = ref('')
const storageError = ref('')
const dataWarnings = ref<string[]>([])
const lastUpdated = ref<string | null>(null)

// 认证状态
const isAuthenticated = ref(false)
const authLoading = ref(true)
const loginEmail = ref('')
const loginPassword = ref('')
const loginError = ref('')
const loginSubmitting = ref(false)
const dashboardStarted = ref(false)

const taskRows = ref<ProcessingTask[]>([])
const workerRows = ref<WorkerNode[]>([])
const modelAssetCount = ref(0)
const memoryPoseCount = ref(0)
const userSummaries = ref<UserActivitySummary[]>([])
const logDrawerVisible = ref(false)
const selectedTaskId = ref<string | null>(null)

const selectedStatus = ref<'all' | TaskStatus>('all')
const selectedTaskType = ref('all')
const searchKeyword = ref('')
const autoRefresh = ref(true)
const refreshSeconds = ref(60)
const taskTrendRange = ref<'24h' | '7d' | '30d' | 'all'>('all')
const isDarkTheme = ref(true)
const accentColor = ref('#6b7a8f')

const THEME_STORAGE_KEY = 'dashboard-theme-dark'
const ACCENT_STORAGE_KEY = 'dashboard-theme-accent'

const channelState = ref<Record<string, string>>({
  processing_tasks: 'connecting',
  model_assets: 'connecting',
  memory_poses: 'connecting',
  worker_nodes: 'connecting',
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
  totalUsers: 0,
  activeUsers24h: 0,
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

const isWorkerRowOnline = (worker: WorkerNode) => {
  const timeout = Number(worker.metadata?.online_timeout_seconds ?? 30)
  return dayjs().diff(dayjs(worker.last_heartbeat), 'second') <= timeout && worker.status !== 'offline'
}

const onlineWorkerRows = computed(() => workerRows.value.filter((item) => isWorkerRowOnline(item)))
const onlineWorkerCount = computed(() => onlineWorkerRows.value.length)
const workerOnline = computed(() => onlineWorkerCount.value > 0)
const workerSummaryText = computed(() => {
  if (!workerRows.value.length) return '0 / 0'
  return `${onlineWorkerCount.value} / ${workerRows.value.length}`
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

const taskInsightMap = computed<Record<string, TaskInsight>>(() =>
  Object.fromEntries(
    taskRows.value.map((task) => [
      task.id,
      buildTaskInsight({
        logs: task.logs,
        status: task.status,
      }),
    ]),
  ),
)

const taskWorkerMap = computed<Record<string, WorkerNode>>(() =>
  Object.fromEntries(
    workerRows.value
      .filter((worker): worker is WorkerNode & { current_task_id: string } => Boolean(worker.current_task_id))
      .map((worker) => [worker.current_task_id, worker]),
  ),
)

const selectedTask = computed(() => taskRows.value.find((item) => item.id === selectedTaskId.value) ?? null)
const selectedTaskInsight = computed(() =>
  selectedTask.value ? taskInsightMap.value[selectedTask.value.id] ?? null : null,
)
const selectedTaskWorker = computed(() =>
  selectedTask.value ? taskWorkerMap.value[selectedTask.value.id] ?? null : null,
)
const selectedTaskWorkerLabel = computed(() =>
  selectedTaskWorker.value ? formatWorkerLabel(selectedTaskWorker.value) : '',
)

const taskQueue = computed(() => filteredTasks.value.slice(0, 20))
const failedTasks = computed(() => taskRows.value.filter((item) => item.status === 'failed').slice(0, 6))
const processingSpotlights = computed(() =>
  filteredTasks.value
    .filter((item) => item.status === 'processing')
    .map((task) => ({
      task,
      insight:
        taskInsightMap.value[task.id] ??
        buildTaskInsight({
          logs: task.logs,
          status: task.status,
        }),
      worker: taskWorkerMap.value[task.id] ?? null,
    }))
    .slice(0, 6),
)
const topUsers = computed(() => userSummaries.value.slice(0, 6))
const newlyActiveUsers = computed(() => userSummaries.value.filter((item) => item.task7d > 0 || item.asset7d > 0).slice(0, 8))

const totalStorageBytes = computed(() =>
  storageStats.value.reduce((sum, item) => sum + item.totalBytes, 0),
)

const edgeHealthyCount = computed(() => edgeChecks.value.filter((item) => item.status === 'ok').length)
const queueCount = computed(() => pendingCount.value + processingCount.value)

const refreshModeText = computed(() => (autoRefresh.value ? `${refreshSeconds.value}s 自动刷新` : '手动刷新'))
const totalUserCount = computed(() => userSummaries.value.length)
const resourceRecordCount = computed(() => modelAssetCount.value + memoryPoseCount.value)

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
  pie: ['#a0aab5', accentColor.value, '#6d8260', '#8b4747'],
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
      emphasis: { itemStyle: { color: '#8393a8' } },
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
        itemStyle: { color: '#e4e8ed', borderColor: accentColor.value, borderWidth: 2 },
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
const formatWorkerLabel = (worker: WorkerNode) => worker.hostname || worker.worker_id
const getWorkerStatusTag = (status: string) => {
  if (status === 'idle') return 'success'
  if (status === 'busy') return 'warning'
  if (status === 'stopping') return 'danger'
  if (status === 'offline') return 'info'
  return 'info'
}
const getWorkerStatusLabel = (worker: WorkerNode) => {
  if (isWorkerRowOnline(worker)) {
    if (worker.status === 'busy') return '执行中'
    if (worker.status === 'stopping') return '停止中'
    if (worker.status === 'idle') return '空闲'
    return '在线'
  }
  return worker.status === 'offline' ? '已离线' : '失联'
}
const formatHeartbeatAge = (iso: string) => {
  const diffSeconds = dayjs().diff(dayjs(iso), 'second')
  if (diffSeconds < 60) return `${diffSeconds}s 前`
  if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m 前`
  return `${Math.floor(diffSeconds / 3600)}h 前`
}

const getLatestLogMessage = (logs: unknown) => {
  return buildTaskInsight({ logs, status: 'processing' }).latestMessage.slice(0, 100)
}

const getTaskInsight = (task: ProcessingTask) =>
  taskInsightMap.value[task.id] ??
  buildTaskInsight({
    logs: task.logs,
    status: task.status,
  })

const getTaskProgress = (task: ProcessingTask) => getTaskInsight(task).percent

const openTaskLogDrawer = (task: ProcessingTask) => {
  selectedTaskId.value = task.id
  logDrawerVisible.value = true
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

const handleLogin = async () => {
  loginError.value = ''
  if (!loginEmail.value || !loginPassword.value) {
    loginError.value = '请输入邮箱和密码'
    return
  }
  loginSubmitting.value = true
  const { error } = await supabase.auth.signInWithPassword({
    email: loginEmail.value,
    password: loginPassword.value,
  })
  loginSubmitting.value = false
  if (error) {
    loginError.value = error.message
  }
}

const handleLogout = async () => {
  await supabase.auth.signOut()
}

const updateWorkerDesiredState = async (
  worker: WorkerNode,
  desiredState: 'run' | 'pause' | 'interrupt',
  successMessage: string,
) => {
  const { error } = await supabase
    .from('worker_nodes')
    .update({
      desired_state: desiredState,
      control_note:
        desiredState === 'run'
          ? 'Resumed from dashboard'
          : desiredState === 'interrupt'
            ? 'Interrupted from dashboard'
            : 'Paused from dashboard',
      control_requested_at: new Date().toISOString(),
    })
    .eq('worker_id', worker.worker_id)

  if (error) {
    ElMessage.error(`控制失败: ${error.message}`)
    return
  }

  ElMessage.success(successMessage)
  await refreshDashboard()
}

const requestWorkerPause = async (worker: WorkerNode) =>
  updateWorkerDesiredState(
    worker,
    'pause',
    `已请求优雅暂停 ${formatWorkerLabel(worker)}，它会停止接新任务并在安全点退出。`,
  )

const requestWorkerInterrupt = async (worker: WorkerNode) =>
  updateWorkerDesiredState(
    worker,
    'interrupt',
    `已请求中断 ${formatWorkerLabel(worker)}，Supervisor 会尝试立即打断当前任务。`,
  )

const requestWorkerResume = async (worker: WorkerNode) =>
  updateWorkerDesiredState(
    worker,
    'run',
    `已请求恢复 ${formatWorkerLabel(worker)}，Supervisor 会在轮询周期内拉起实例。`,
  )

const edgeFunctionNames = computed(() => {
  const raw = (import.meta.env.VITE_SUPABASE_EDGE_FUNCTIONS as string | undefined) ?? ''
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
})

const edgeStatusText = computed(() => {
  if (!edgeFunctionNames.value.length) return '未配置探测'
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
    label: 'Workers',
    value: workerSummaryText.value,
    note: workerOnline.value ? '心跳正常' : '无在线实例',
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
    key: 'users',
    label: '用户总数',
    value: `${totalUserCount.value}`,
    note: `24h 活跃 ${timeBasedStats.value.activeUsers24h} / 7d 活跃 ${timeBasedStats.value.activeUsers7d}`,
    icon: 'lucide:users',
    tone: 'neutral' as const,
  },
  {
    key: 'assets',
    label: '资源数据',
    value: `${resourceRecordCount.value}`,
    note: `模型 ${modelAssetCount.value} / 姿态 ${memoryPoseCount.value}`,
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

const formatUserId = (userId: string) => {
  if (userId.length <= 14) return userId
  return `${userId.slice(0, 8)}...${userId.slice(-4)}`
}

const buildUserSummaries = (rows: UserActivityAggregateRow[]): UserActivitySummary[] =>
  rows.map((s) => ({
    userId: s.user_id,
    displayName: s.user_id.slice(0, 8) + '...',
    taskCount: s.total_tasks,
    assetCount: s.total_assets,
    task24h: s.tasks_24h,
    task7d: s.tasks_7d,
    asset7d: s.assets_7d,
    lastSeenAt: s.last_active,
  }))

const fetchUserActivitySummary = async (since24h: string, since7d: string) => {
  const [taskUserRes, assetUserRes] = await Promise.all([
    supabase
      .from('processing_tasks')
      .select('user_id, created_at')
      .order('created_at', { ascending: false })
      .limit(5000),
    supabase
      .from('model_assets')
      .select('user_id, created_at')
      .order('created_at', { ascending: false })
      .limit(5000),
  ])

  if (taskUserRes.error || assetUserRes.error) {
    dataWarnings.value.push(
      `用户活跃聚合失败：${taskUserRes.error?.message || assetUserRes.error?.message || '未知错误'}`,
    )
    return []
  }

  const byUser = new Map<string, UserActivityAggregateRow>()
  const ensureUser = (userId: string) => {
    const existing = byUser.get(userId)
    if (existing) return existing
    const created: UserActivityAggregateRow = {
      user_id: userId,
      total_tasks: 0,
      tasks_24h: 0,
      tasks_7d: 0,
      total_assets: 0,
      assets_7d: 0,
      last_active: null,
    }
    byUser.set(userId, created)
    return created
  }

  const touchLastActive = (row: UserActivityAggregateRow, createdAt: string | null) => {
    if (!createdAt) return
    if (!row.last_active || dayjs(createdAt).isAfter(dayjs(row.last_active))) {
      row.last_active = createdAt
    }
  }

  ;((taskUserRes.data ?? []) as Array<{ user_id: string | null; created_at: string | null }>).forEach((item) => {
    if (!item.user_id) return
    const row = ensureUser(item.user_id)
    row.total_tasks += 1
    if (item.created_at && item.created_at >= since24h) row.tasks_24h += 1
    if (item.created_at && item.created_at >= since7d) row.tasks_7d += 1
    touchLastActive(row, item.created_at)
  })

  ;((assetUserRes.data ?? []) as Array<{ user_id: string | null; created_at: string | null }>).forEach((item) => {
    if (!item.user_id) return
    const row = ensureUser(item.user_id)
    row.total_assets += 1
    if (item.created_at && item.created_at >= since7d) row.assets_7d += 1
    touchLastActive(row, item.created_at)
  })

  return Array.from(byUser.values()).sort((a, b) => {
    const left = a.last_active ? dayjs(a.last_active).valueOf() : 0
    const right = b.last_active ? dayjs(b.last_active).valueOf() : 0
    return right - left
  })
}

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
  if (!edgeFunctionNames.value.length) {
    edgeChecks.value = []
    edgeLoading.value = false
    return
  }

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
  dataWarnings.value = []

  const now = dayjs()
  const since24h = now.subtract(24, 'hour').toISOString()
  const since7d = now.subtract(7, 'day').toISOString()

  const [
    tasksRes,
    workerRes,
    processingTaskCountRes,
    assetCountRes,
    poseCountRes,
    ragCountRes,
    taskTableCountRes,
    task24hRes,
    asset7dRes,
  ] = await Promise.all([
    supabase
      .from('processing_tasks')
      .select('id, display_name, scene_id, user_id, status, task_type, quality_score, created_at, updated_at, logs')
      .order('updated_at', { ascending: false })
      .limit(500),
    supabase
      .from('worker_nodes')
      .select('worker_id, hostname, pid, status, current_task_id, current_scene_id, desired_state, control_note, last_heartbeat, started_at, stopped_at, metadata')
      .order('last_heartbeat', { ascending: false })
      .limit(100),
    supabase.from('processing_tasks').select('*', { count: 'exact', head: true }),
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

  const readErrors = [
    tasksRes.error ? `processing_tasks 列表读取失败：${tasksRes.error.message}` : '',
    workerRes.error ? `worker_nodes 读取失败：${workerRes.error.message}` : '',
    processingTaskCountRes.error ? `processing_tasks 计数失败：${processingTaskCountRes.error.message}` : '',
    assetCountRes.error ? `model_assets 计数失败：${assetCountRes.error.message}` : '',
    poseCountRes.error ? `memory_poses 计数失败：${poseCountRes.error.message}` : '',
  ].filter(Boolean)

  dataWarnings.value.push(
    ...readErrors,
    ...(ragCountRes.error ? [`rag_docs 计数失败：${ragCountRes.error.message}`] : []),
    ...(taskTableCountRes.error ? [`tasks 计数失败：${taskTableCountRes.error.message}`] : []),
    ...(task24hRes.error ? [`24h 任务统计失败：${task24hRes.error.message}`] : []),
    ...(asset7dRes.error ? [`7d 资产统计失败：${asset7dRes.error.message}`] : []),
  )

  if (readErrors.length) {
    errorMessage.value = '部分数据读取失败，已保留可读取的数据。'
  }

  const tasks = tasksRes.error ? [] : ((tasksRes.data ?? []) as ProcessingTask[])
  const workers = workerRes.error ? [] : ((workerRes.data ?? []) as WorkerNode[])
  const activityRows = await fetchUserActivitySummary(since24h, since7d)

  taskRows.value = tasks
  workerRows.value = workers
  userSummaries.value = buildUserSummaries(activityRows)
  modelAssetCount.value = assetCountRes.error ? 0 : (assetCountRes.count ?? 0)
  memoryPoseCount.value = poseCountRes.error ? 0 : (poseCountRes.count ?? 0)

  const tasks24hRows = task24hRes.error
    ? []
    : ((task24hRes.data ?? []) as Array<{ status: string; user_id: string }>)
  timeBasedStats.value = {
    tasks24h: task24hRes.error ? 0 : (task24hRes.count ?? 0),
    failed24h: tasks24hRows.filter((item) => item.status === 'failed').length,
    completed24h: tasks24hRows.filter((item) => item.status === 'completed').length,
    totalUsers: userSummaries.value.length,
    activeUsers24h: new Set(tasks24hRows.map((item) => item.user_id)).size,
    activeUsers7d: userSummaries.value.filter((item) => item.task7d > 0 || item.asset7d > 0).length,
    assets7d: asset7dRes.error ? 0 : (asset7dRes.count ?? 0),
  }

  dbCounts.value = {
    processing_tasks: processingTaskCountRes.error ? 0 : (processingTaskCountRes.count ?? 0),
    model_assets: assetCountRes.error ? 0 : (assetCountRes.count ?? 0),
    memory_poses: poseCountRes.error ? 0 : (poseCountRes.count ?? 0),
    rag_docs: ragCountRes.error ? 0 : (ragCountRes.count ?? 0),
    tasks: taskTableCountRes.error ? 0 : (taskTableCountRes.count ?? 0),
  }

  lastUpdated.value = dayjs().format('YYYY-MM-DD HH:mm:ss')

  await Promise.all([fetchStorageStats(), refreshEdgeChecks()])

  loading.value = false
  refreshing.value = false
}

let refreshTimer: number | undefined
let pollTimer: number | undefined
const channels: RealtimeChannel[] = []

const clearDashboardRuntime = () => {
  if (refreshTimer) {
    window.clearTimeout(refreshTimer)
    refreshTimer = undefined
  }
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = undefined
  }
  channels.splice(0).forEach((channel) => {
    void supabase.removeChannel(channel)
  })
  dashboardStarted.value = false
  channelState.value = {
    processing_tasks: 'connecting',
    model_assets: 'connecting',
    memory_poses: 'connecting',
    worker_nodes: 'connecting',
  }
}

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

const startDashboardRuntime = async () => {
  if (dashboardStarted.value) return
  dashboardStarted.value = true
  loading.value = true
  await refreshDashboard()

  bindChannel('processing_tasks', 'dashboard-processing-tasks')
  bindChannel('model_assets', 'dashboard-model-assets')
  bindChannel('memory_poses', 'dashboard-memory-poses')
  bindChannel('worker_nodes', 'dashboard-worker-nodes')

  restartPolling()
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

  // 必须等待 getSession 完成后再启动数据刷新，否则首次加载会停在空面板。
  const { data: { session } } = await supabase.auth.getSession()
  isAuthenticated.value = !!session
  authLoading.value = false

  supabase.auth.onAuthStateChange((_event, nextSession) => {
    const authed = !!nextSession
    isAuthenticated.value = authed
    authLoading.value = false

    if (authed) {
      void startDashboardRuntime()
    } else {
      clearDashboardRuntime()
      loading.value = true
    }
  })

  if (session) await startDashboardRuntime()
})

watch([isDarkTheme, accentColor], () => {
  localStorage.setItem(THEME_STORAGE_KEY, isDarkTheme.value ? '1' : '0')
  localStorage.setItem(ACCENT_STORAGE_KEY, accentColor.value)
  applyTheme()
})

onUnmounted(() => {
  clearDashboardRuntime()
})
</script>

<template>
  <!-- 认证加载中 -->
  <div v-if="authLoading" class="auth-loading">
    <p>正在验证登录状态...</p>
  </div>

  <!-- 登录表单 -->
  <div v-else-if="!isAuthenticated" class="login-page">
    <div class="login-card glass-card">
      <h2>BrainDance Dashboard</h2>
      <p class="login-subtitle">请登录以访问管理面板</p>
      <form @submit.prevent="handleLogin">
        <el-input
          v-model="loginEmail"
          type="email"
          placeholder="邮箱"
          :disabled="loginSubmitting"
          style="margin-bottom: 12px;"
        />
        <el-input
          v-model="loginPassword"
          type="password"
          placeholder="密码"
          show-password
          :disabled="loginSubmitting"
          style="margin-bottom: 12px;"
        />
        <p v-if="loginError" class="login-error">{{ loginError }}</p>
        <el-button
          type="primary"
          native-type="submit"
          :loading="loginSubmitting"
          style="width: 100%;"
        >
          登录
        </el-button>
      </form>
    </div>
  </div>

  <!-- 已认证：主 Dashboard -->
  <div v-else class="dashboard-page">
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
              <span>资源</span>
              <strong>{{ resourceRecordCount }}</strong>
            </article>
            <article class="phone-stat-card">
              <span>用户</span>
              <strong>{{ totalUserCount }}</strong>
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
                  :predefine="['#6b7a8f', '#71839a', '#6d8260', '#8b4747', '#a0aab5']"
                />
              </div>

              <el-button size="small" @click="handleLogout">
                <Icon icon="lucide:log-out" />
                <span>登出</span>
              </el-button>
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
              <span>Workers</span>
              <strong>{{ workerSummaryText }}</strong>
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
        <el-alert
          v-for="warning in dataWarnings"
          :key="warning"
          :title="warning"
          type="warning"
          show-icon
          class="mb-16"
        />
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

          <section class="panel-grid panel-grid--processing">
            <el-card shadow="never" class="table-card glass-card">
              <template #header>
                <div class="card-header-row">
                  <div>
                    <div class="card-header">处理中模型</div>
                    <div class="header-meta">按日志阶段估算进度，优先展示正在跑的任务。</div>
                  </div>
                  <div class="filter-count">{{ processingSpotlights.length }} 条</div>
                </div>
              </template>

              <el-empty
                v-if="!processingSpotlights.length"
                description="当前没有 processing 状态的模型任务"
              />

              <div v-else class="processing-grid">
                <article
                  v-for="item in processingSpotlights"
                  :key="item.task.id"
                  class="processing-card"
                >
                  <div class="processing-card__top">
                    <div>
                      <div class="task-name">{{ formatDisplayName(item.task) }}</div>
                      <div class="task-sub">
                        {{ item.task.task_type || 'video_3dgs' }} / {{ item.task.scene_id }}
                      </div>
                    </div>
                    <el-tag type="warning">{{ item.insight.stageLabel }}</el-tag>
                  </div>

                  <div class="processing-card__metrics">
                    <div class="processing-chip">
                      <span>当前 Worker</span>
                      <strong>{{ item.worker ? formatWorkerLabel(item.worker) : '待分配' }}</strong>
                    </div>
                    <div class="processing-chip">
                      <span>最新心跳</span>
                      <strong>{{ item.worker ? formatHeartbeatAge(item.worker.last_heartbeat) : '暂无' }}</strong>
                    </div>
                  </div>

                  <el-progress :percentage="item.insight.percent" :stroke-width="10" />

                  <div class="processing-card__log">
                    <span>最新动态</span>
                    <strong>{{ item.insight.latestMessage }}</strong>
                  </div>

                  <div class="processing-card__footer">
                    <span>更新时间 {{ formatDateTime(item.task.updated_at) }}</span>
                    <el-button size="small" plain @click="openTaskLogDrawer(item.task)">查看日志</el-button>
                  </div>
                </article>
              </div>
            </el-card>

            <el-card shadow="never" class="fail-card glass-card">
              <template #header>
                <div>
                  <div class="card-header">处理观察点</div>
                  <div class="header-meta">聚焦处理阶段、排障入口和 Worker 绑定情况。</div>
                </div>
              </template>

              <div class="alerts-list alerts-list--soft">
                <article
                  v-for="item in processingSpotlights.slice(0, 4)"
                  :key="item.task.id"
                  class="alert-item"
                >
                  <div class="alert-item-top">
                    <div>
                      <span class="alert-label">{{ item.insight.stageLabel }}</span>
                      <strong class="alert-value">{{ item.insight.percent }}%</strong>
                    </div>
                    <el-tag type="info">
                      {{ item.worker ? getWorkerStatusLabel(item.worker) : '等待接单' }}
                    </el-tag>
                  </div>
                  <p class="alert-note">{{ item.insight.summary }}</p>
                  <div class="fail-sub">
                    {{ item.worker ? formatWorkerLabel(item.worker) : '尚未绑定 Worker' }}
                  </div>
                </article>
              </div>

              <el-divider />

              <div class="db-metrics db-metrics--users">
                <div class="metric-item">
                  <div class="metric-title">处理中</div>
                  <div class="metric-value">{{ processingCount }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">排队中</div>
                  <div class="metric-value">{{ pendingCount }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">在线 Worker</div>
                  <div class="metric-value ok">{{ onlineWorkerCount }}</div>
                </div>
              </div>
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
                    <div class="task-progress-cell">
                      <el-progress
                        :percentage="getTaskProgress(scope.row)"
                        :status="scope.row.status === 'failed' ? 'exception' : scope.row.status === 'completed' ? 'success' : undefined"
                      />
                      <span class="task-progress-note">{{ getTaskInsight(scope.row).stageLabel }}</span>
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="日志摘要" min-width="240">
                  <template #default="scope">
                    <div class="task-log-snippet">{{ getTaskInsight(scope.row).latestMessage }}</div>
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
                <el-table-column label="日志" width="110" align="center">
                  <template #default="scope">
                    <el-button size="small" plain @click="openTaskLogDrawer(scope.row)">查看</el-button>
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
                  <div style="margin-top: 10px;">
                    <el-button size="small" plain @click="openTaskLogDrawer(item)">查看完整日志</el-button>
                  </div>
                </el-timeline-item>
              </el-timeline>
            </el-card>
          </section>

          <section class="panel-grid panel-grid--resources">
            <el-card shadow="never" class="table-card glass-card">
              <template #header>
                <div class="card-header-row">
                  <div>
                    <div class="card-header">Worker 集群</div>
                    <div class="header-meta">在线 {{ onlineWorkerCount }} / 总数 {{ workerRows.length }}，支持对单个实例发起优雅暂停。</div>
                  </div>
                </div>
              </template>
              <el-table :data="workerRows" stripe height="300" empty-text="还没有 worker 注册心跳">
                <el-table-column label="Worker" min-width="240">
                  <template #default="scope">
                    <div class="task-name">{{ formatWorkerLabel(scope.row) }}</div>
                    <div class="task-sub">{{ scope.row.worker_id }}</div>
                  </template>
                </el-table-column>
                <el-table-column label="状态" width="120" align="center">
                  <template #default="scope">
                    <el-tag :type="getWorkerStatusTag(scope.row.status)">
                      {{ getWorkerStatusLabel(scope.row) }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="当前任务" min-width="180">
                  <template #default="scope">
                    {{ scope.row.current_scene_id || scope.row.current_task_id || '-' }}
                  </template>
                </el-table-column>
                <el-table-column label="心跳" min-width="170">
                  <template #default="scope">
                    {{ formatDateTime(scope.row.last_heartbeat) }} / {{ formatHeartbeatAge(scope.row.last_heartbeat) }}
                  </template>
                </el-table-column>
                <el-table-column label="控制" min-width="260" align="center">
                  <template #default="scope">
                    <div style="display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;">
                      <el-button
                        size="small"
                        plain
                        :disabled="scope.row.desired_state === 'pause' || !isWorkerRowOnline(scope.row)"
                        @click="requestWorkerPause(scope.row)"
                      >
                        优雅暂停
                      </el-button>
                      <el-button
                        size="small"
                        type="danger"
                        plain
                        :disabled="scope.row.desired_state === 'interrupt' || !isWorkerRowOnline(scope.row)"
                        @click="requestWorkerInterrupt(scope.row)"
                      >
                        中断任务
                      </el-button>
                      <el-button
                        size="small"
                        type="success"
                        plain
                        :disabled="scope.row.desired_state === 'run' && isWorkerRowOnline(scope.row)"
                        @click="requestWorkerResume(scope.row)"
                      >
                        恢复实例
                      </el-button>
                    </div>
                  </template>
                </el-table-column>
              </el-table>
              <div class="header-meta" style="margin-top: 12px;">
                “优雅暂停” 会把 `desired_state` 设为 `pause`，实例不会再接新任务；“中断任务” 会把 `desired_state` 设为 `interrupt`，Supervisor 会尝试向子 Worker 转发中断信号，尽量打断当前任务；“恢复实例” 会把 `desired_state` 改回 `run` 并重新拉起实例。
              </div>
            </el-card>

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
                  <div class="metric-title">用户总数</div>
                  <div class="metric-value">{{ timeBasedStats.totalUsers }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">24h 活跃用户</div>
                  <div class="metric-value">{{ timeBasedStats.activeUsers24h }}</div>
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

          <section class="panel-grid panel-grid--users">
            <el-card shadow="never" class="table-card glass-card">
              <template #header>
                <div>
                  <div class="card-header">用户列表</div>
                  <div class="header-meta">基于 `processing_tasks`、`model_assets`、`tasks` 的 user_id 聚合。</div>
                </div>
              </template>
              <el-table :data="userSummaries" stripe height="360" empty-text="暂无用户数据">
                <el-table-column label="用户" min-width="180">
                  <template #default="scope">
                    <div class="task-name">{{ formatUserId(scope.row.userId) }}</div>
                    <div class="task-sub">{{ scope.row.userId }}</div>
                  </template>
                </el-table-column>
                <el-table-column label="任务数" width="90" align="right" prop="taskCount" />
                <el-table-column label="资产数" width="90" align="right" prop="assetCount" />
                <el-table-column label="24h 任务" width="100" align="right" prop="task24h" />
                <el-table-column label="7d 活跃" width="100" align="center">
                  <template #default="scope">
                    <el-tag :type="scope.row.task7d > 0 || scope.row.asset7d > 0 ? 'success' : 'info'">
                      {{ scope.row.task7d > 0 || scope.row.asset7d > 0 ? '活跃' : '沉默' }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="最近活动" min-width="170">
                  <template #default="scope">
                    {{ scope.row.lastSeenAt ? formatDateTime(scope.row.lastSeenAt) : '-' }}
                  </template>
                </el-table-column>
              </el-table>
            </el-card>

            <el-card shadow="never" class="fail-card glass-card">
              <template #header>
                <div>
                  <div class="card-header">用户活跃摘要</div>
                  <div class="header-meta">最近活跃与高频使用者。</div>
                </div>
              </template>

              <div class="db-metrics db-metrics--users">
                <div class="metric-item">
                  <div class="metric-title">总用户</div>
                  <div class="metric-value">{{ timeBasedStats.totalUsers }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">24h 活跃</div>
                  <div class="metric-value">{{ timeBasedStats.activeUsers24h }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-title">7d 活跃</div>
                  <div class="metric-value ok">{{ timeBasedStats.activeUsers7d }}</div>
                </div>
              </div>

              <div class="user-summary-block">
                <div class="card-header-row user-summary-block__head">
                  <div>
                    <div class="card-header">Top 用户</div>
                    <div class="header-meta">按业务动作总数排序。</div>
                  </div>
                </div>
                <div class="alerts-list alerts-list--soft">
                  <article v-for="item in topUsers" :key="item.userId" class="alert-item">
                    <div class="alert-item-top">
                      <div>
                        <span class="alert-label">{{ formatUserId(item.userId) }}</span>
                        <strong class="alert-value">{{ item.taskCount + item.assetCount }}</strong>
                      </div>
                      <el-tag type="info">任务 {{ item.taskCount }}</el-tag>
                    </div>
                    <p class="alert-note">资产 {{ item.assetCount }}，最近活动 {{ item.lastSeenAt ? formatDateTime(item.lastSeenAt) : '-' }}</p>
                  </article>
                </div>
              </div>

              <el-divider />

              <div class="user-chip-list">
                <div v-for="item in newlyActiveUsers" :key="item.userId" class="user-chip">
                  <span>{{ formatUserId(item.userId) }}</span>
                  <strong>{{ item.task7d + item.asset7d }}</strong>
                </div>
              </div>
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

    <task-log-drawer
      v-model="logDrawerVisible"
      :task="selectedTask"
      :insight="selectedTaskInsight"
      :worker-label="selectedTaskWorkerLabel"
    />
  </div>
</template>
