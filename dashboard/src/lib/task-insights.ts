export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | string

export interface TaskLogEntry {
  id: string
  message: string
  stage: string
  stageLabel: string
  level: 'info' | 'success' | 'warning' | 'danger'
  icon: string
  percent: number
  timestamp: number | null
  timeLabel: string
}

export interface TaskInsight {
  percent: number
  stage: string
  stageLabel: string
  headline: string
  latestMessage: string
  summary: string
  level: TaskLogEntry['level']
  logEntries: TaskLogEntry[]
}

interface RawLogObject {
  ts?: number | string | null
  msg?: string | null
  message?: string | null
  level?: string | null
}

const STAGE_META = [
  {
    key: 'pending',
    label: '等待调度',
    percent: 8,
    icon: 'lucide:hourglass',
    keywords: ['排队', '等待', 'pending'],
  },
  {
    key: 'download',
    label: '准备素材',
    percent: 18,
    icon: 'lucide:cloud-download',
    keywords: ['下载', '云端', '资源', 'raw/', '素材'],
  },
  {
    key: 'pipeline',
    label: '装载流水线',
    percent: 26,
    icon: 'lucide:workflow',
    keywords: ['加载流水线', '最佳帧', '候选帧', '快链', '慢链', '目标判定'],
  },
  {
    key: 'preprocess',
    label: '预处理',
    percent: 38,
    icon: 'lucide:clapperboard',
    keywords: ['抽帧', 'ffmpeg', '图片准备', 'frames', '预处理'],
  },
  {
    key: 'analyze',
    label: '分析质检',
    percent: 48,
    icon: 'lucide:scan-search',
    keywords: ['质检', 'rag', '语义分析', '标签', 'embedding', '最佳预览'],
  },
  {
    key: 'reconstruct',
    label: '位姿解算',
    percent: 62,
    icon: 'lucide:map-pinned',
    keywords: ['位姿', 'colmap', 'glomap', 'da3', '匹配率', 'mapper'],
  },
  {
    key: 'segment',
    label: '语义分割',
    percent: 72,
    icon: 'lucide:scissors-line-dashed',
    keywords: ['分割', 'mask', 'sam', '抠图'],
  },
  {
    key: 'train',
    label: '模型生成',
    percent: 84,
    icon: 'lucide:sparkles',
    keywords: ['训练', '导出', 'point_cloud', 'sparse2dgs', 'sugar', '3dgs', '反投影', '生成'],
  },
  {
    key: 'publish',
    label: '上传交付',
    percent: 94,
    icon: 'lucide:upload',
    keywords: ['上传', 'preview', 'transforms', '知识库', '压缩完成', 'model_assets'],
  },
  {
    key: 'completed',
    label: '处理完成',
    percent: 100,
    icon: 'lucide:badge-check',
    keywords: ['任务全部完成', '完成', '成功', '已上传到 supabase'],
  },
]

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const normalizeTimestamp = (value: number | string | null | undefined) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value > 1e12 ? value : value * 1000
  }

  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) {
      return parsed > 1e12 ? parsed : parsed * 1000
    }
  }

  return null
}

const formatTimeLabel = (timestamp: number | null) => {
  if (!timestamp) return '刚刚'
  return new Date(timestamp).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

const inferLevel = (message: string, rawLevel?: string | null): TaskLogEntry['level'] => {
  const level = (rawLevel || '').toLowerCase()
  const lower = message.toLowerCase()

  if (level.includes('error') || /❌|失败|异常|错误|fatal/.test(message) || lower.includes('error')) return 'danger'
  if (level.includes('warn') || /⚠️|警告|回退|跳过/.test(message)) return 'warning'
  if (level.includes('success') || /✅|完成|成功/.test(message)) return 'success'
  return 'info'
}

const inferStageFromMessage = (message: string) => {
  const lower = message.toLowerCase()

  for (const stage of STAGE_META) {
    if (stage.keywords.some((keyword) => lower.includes(keyword.toLowerCase()))) {
      return stage
    }
  }

  return STAGE_META[1]!
}

const inferPercentFromMessage = (message: string, fallback: number) => {
  const percentMatch = message.match(/(\d{1,3})\s*%/)
  if (percentMatch) {
    const numeric = Number(percentMatch[1])
    if (Number.isFinite(numeric)) return clamp(numeric, 0, 100)
  }

  const fractionMatch = message.match(/(\d+)\s*\/\s*(\d+)/)
  if (fractionMatch) {
    const current = Number(fractionMatch[1])
    const total = Number(fractionMatch[2])
    if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
      return clamp(Math.round((current / total) * 100), fallback, 99)
    }
  }

  return fallback
}

export const normalizeTaskLogs = (logs: unknown): TaskLogEntry[] => {
  if (!Array.isArray(logs)) return []

  return logs
    .map((raw, index) => {
      const objectEntry = (raw && typeof raw === 'object' ? raw : null) as RawLogObject | null
      const message =
        typeof raw === 'string'
          ? raw
          : objectEntry?.msg || objectEntry?.message || ''

      if (!message.trim()) return null

      const stage = inferStageFromMessage(message)
      const timestamp = normalizeTimestamp(objectEntry?.ts)
      const level = inferLevel(message, objectEntry?.level)

      return {
        id: `${timestamp ?? 'log'}-${index}`,
        message,
        stage: stage.key,
        stageLabel: stage.label,
        level,
        icon: stage.icon,
        percent: inferPercentFromMessage(message, stage.percent),
        timestamp,
        timeLabel: formatTimeLabel(timestamp),
      } satisfies TaskLogEntry
    })
    .filter((entry): entry is TaskLogEntry => Boolean(entry))
    .sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
}

export const buildTaskInsight = ({
  logs,
  status,
}: {
  logs: unknown
  status: TaskStatus
}): TaskInsight => {
  const entries = normalizeTaskLogs(logs)
  const latest = entries.length ? entries[entries.length - 1] : undefined

  if (status === 'completed') {
    return {
      percent: 100,
      stage: 'completed',
      stageLabel: '处理完成',
      headline: latest?.message || '任务已完成',
      latestMessage: latest?.message || '任务已完成',
      summary: entries.length ? `共记录 ${entries.length} 条关键日志` : '暂无关键日志回传',
      level: 'success',
      logEntries: entries,
    }
  }

  if (status === 'failed') {
    const failureEntry = [...entries].reverse().find((item) => item.level === 'danger') || latest
    return {
      percent: clamp(failureEntry?.percent ?? 86, 10, 99),
      stage: failureEntry?.stage || 'failed',
      stageLabel: failureEntry?.stageLabel || '执行失败',
      headline: failureEntry?.message || '任务执行失败',
      latestMessage: failureEntry?.message || '任务执行失败',
      summary: entries.length ? '请优先查看最后一条异常日志' : '失败时未回传关键日志',
      level: 'danger',
      logEntries: entries,
    }
  }

  if (status === 'pending') {
    return {
      percent: 8,
      stage: 'pending',
      stageLabel: '等待调度',
      headline: latest?.message || '等待 Worker 接单',
      latestMessage: latest?.message || '等待 Worker 接单',
      summary: entries.length ? `已记录 ${entries.length} 条前置信息` : '队列中，尚未开始执行',
      level: 'warning',
      logEntries: entries,
    }
  }

  const processingEntry = latest || {
    stage: 'download',
    stageLabel: '准备素材',
    percent: 18,
    message: '任务已开始，等待第一条关键日志',
    level: 'info' as const,
  }

  const maxPercent = entries.reduce((max, entry) => Math.max(max, entry.percent), processingEntry.percent)

  return {
    percent: clamp(maxPercent, 12, 99),
    stage: processingEntry.stage,
    stageLabel: processingEntry.stageLabel,
    headline: processingEntry.message,
    latestMessage: processingEntry.message,
    summary: entries.length ? `当前阶段：${processingEntry.stageLabel}` : '处理中，但日志尚未同步',
    level: processingEntry.level,
    logEntries: entries,
  }
}
