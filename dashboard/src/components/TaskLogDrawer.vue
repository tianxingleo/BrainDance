<script setup lang="ts">
import { computed } from 'vue'
import { Icon } from '@iconify/vue'
import dayjs from 'dayjs'
import type { TaskInsight } from '../lib/task-insights'

interface TaskRecord {
  id: string
  display_name: string | null
  scene_id: string
  user_id: string
  status: string
  task_type: string | null
  created_at: string
  updated_at: string
}

const props = defineProps<{
  modelValue: boolean
  task: TaskRecord | null
  insight: TaskInsight | null
  workerLabel?: string
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
}>()

const visible = computed({
  get: () => props.modelValue,
  set: (value: boolean) => emit('update:modelValue', value),
})

const statusMeta = computed(() => {
  if (props.task?.status === 'completed') return { label: '已完成', type: 'success' as const }
  if (props.task?.status === 'failed') return { label: '失败', type: 'danger' as const }
  if (props.task?.status === 'processing') return { label: '处理中', type: 'warning' as const }
  return { label: '排队中', type: 'info' as const }
})

const durationText = computed(() => {
  if (!props.task) return '-'
  const created = dayjs(props.task.created_at)
  const updated = dayjs(props.task.updated_at)
  const seconds = Math.max(updated.diff(created, 'second'), 0)
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
})

const title = computed(() => props.task?.display_name || props.task?.scene_id || props.task?.id || '任务日志')
</script>

<template>
  <el-drawer
    v-model="visible"
    size="min(720px, 100vw)"
    :with-header="false"
    class="task-log-drawer"
    destroy-on-close
  >
    <div v-if="task && insight" class="drawer-shell">
      <section class="drawer-hero">
        <div>
          <p class="drawer-eyebrow">Task Journal</p>
          <h3>{{ title }}</h3>
          <p class="drawer-copy">{{ insight.headline }}</p>
        </div>
        <div class="drawer-progress">
          <strong>{{ insight.percent }}%</strong>
          <span>{{ insight.stageLabel }}</span>
        </div>
      </section>

      <section class="drawer-meta">
        <article class="meta-card">
          <span>任务状态</span>
          <el-tag :type="statusMeta.type">{{ statusMeta.label }}</el-tag>
        </article>
        <article class="meta-card">
          <span>任务类型</span>
          <strong>{{ task.task_type || 'video_3dgs' }}</strong>
        </article>
        <article class="meta-card">
          <span>Worker</span>
          <strong>{{ workerLabel || '暂未绑定' }}</strong>
        </article>
        <article class="meta-card">
          <span>累计耗时</span>
          <strong>{{ durationText }}</strong>
        </article>
      </section>

      <section class="drawer-bar">
        <div>
          <span>当前阶段</span>
          <strong>{{ insight.summary }}</strong>
        </div>
        <el-progress
          :percentage="insight.percent"
          :status="task.status === 'failed' ? 'exception' : task.status === 'completed' ? 'success' : undefined"
          :stroke-width="10"
        />
      </section>

      <section class="drawer-details">
        <article class="detail-card">
          <span>Scene ID</span>
          <strong>{{ task.scene_id }}</strong>
        </article>
        <article class="detail-card">
          <span>User ID</span>
          <strong>{{ task.user_id }}</strong>
        </article>
        <article class="detail-card">
          <span>创建时间</span>
          <strong>{{ dayjs(task.created_at).format('YYYY-MM-DD HH:mm:ss') }}</strong>
        </article>
        <article class="detail-card">
          <span>最后更新</span>
          <strong>{{ dayjs(task.updated_at).format('YYYY-MM-DD HH:mm:ss') }}</strong>
        </article>
      </section>

      <section class="drawer-timeline">
        <div class="timeline-head">
          <div>
            <p class="drawer-eyebrow">Structured Logs</p>
            <h4>关键日志时间线</h4>
          </div>
          <span class="timeline-count">{{ insight.logEntries.length }} 条</span>
        </div>

        <el-empty
          v-if="!insight.logEntries.length"
          description="数据库里还没有关键日志回传"
        />

        <div v-else class="timeline-list">
          <article
            v-for="entry in insight.logEntries"
            :key="entry.id"
            class="timeline-card"
            :class="`tone-${entry.level}`"
          >
            <div class="timeline-icon">
              <Icon :icon="entry.icon" />
            </div>
            <div class="timeline-body">
              <div class="timeline-top">
                <div>
                  <span class="timeline-stage">{{ entry.stageLabel }}</span>
                  <strong>{{ entry.message }}</strong>
                </div>
                <span class="timeline-time">{{ entry.timeLabel }}</span>
              </div>
              <div class="timeline-progress">
                <span>{{ entry.percent }}%</span>
                <div class="timeline-progress-bar">
                  <i :style="{ width: `${entry.percent}%` }"></i>
                </div>
              </div>
            </div>
          </article>
        </div>
      </section>
    </div>
  </el-drawer>
</template>

<style scoped>
.drawer-shell {
  display: grid;
  gap: 18px;
  height: 100%;
  color: var(--ink-1);
}

.drawer-hero {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 20px;
  border-radius: 26px;
  background:
    radial-gradient(circle at top right, color-mix(in srgb, var(--accent-color) 18%, transparent), transparent 48%),
    linear-gradient(180deg, var(--surface-strong), var(--surface));
  border: 1px solid var(--stroke);
}

.drawer-eyebrow {
  margin: 0 0 8px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.drawer-hero h3,
.timeline-head h4 {
  margin: 0;
}

.drawer-copy {
  margin: 10px 0 0;
  color: var(--ink-2);
  line-height: 1.6;
}

.drawer-progress {
  min-width: 124px;
  padding: 16px 18px;
  border-radius: 22px;
  background: color-mix(in srgb, var(--surface-soft) 86%, transparent);
  border: 1px solid var(--stroke);
  text-align: right;
}

.drawer-progress strong {
  display: block;
  font-size: 2rem;
  line-height: 1;
}

.drawer-progress span,
.meta-card span,
.detail-card span,
.drawer-bar span,
.timeline-stage,
.timeline-time {
  color: var(--muted);
  font-size: 12px;
}

.drawer-meta,
.drawer-details {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.meta-card,
.detail-card,
.drawer-bar,
.timeline-card {
  border-radius: 20px;
  border: 1px solid var(--stroke);
  background: color-mix(in srgb, var(--surface-soft) 86%, transparent);
}

.meta-card,
.detail-card {
  display: grid;
  gap: 8px;
  padding: 14px 16px;
}

.meta-card strong,
.detail-card strong,
.drawer-bar strong,
.timeline-top strong {
  font-size: 0.95rem;
  line-height: 1.5;
}

.drawer-bar {
  display: grid;
  gap: 12px;
  padding: 16px;
}

.drawer-details {
  gap: 10px;
}

.timeline-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: end;
}

.timeline-count {
  padding: 8px 12px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--surface-soft) 86%, transparent);
  border: 1px solid var(--stroke);
  color: var(--ink-2);
  font-size: 12px;
}

.timeline-list {
  display: grid;
  gap: 12px;
  margin-top: 14px;
}

.timeline-card {
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr);
  gap: 12px;
  padding: 14px;
}

.timeline-icon {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.04);
}

.timeline-body {
  display: grid;
  gap: 12px;
}

.timeline-top {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.timeline-stage {
  display: block;
  margin-bottom: 4px;
}

.timeline-time {
  white-space: nowrap;
}

.timeline-progress {
  display: flex;
  align-items: center;
  gap: 10px;
}

.timeline-progress span {
  min-width: 40px;
  color: var(--ink-2);
}

.timeline-progress-bar {
  flex: 1;
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
}

.timeline-progress-bar i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, color-mix(in srgb, var(--accent-color) 70%, white 4%), var(--accent-color));
}

.tone-success .timeline-icon {
  color: var(--ok);
}

.tone-warning .timeline-icon {
  color: var(--warn);
}

.tone-danger .timeline-icon {
  color: var(--bad);
}

@media (max-width: 760px) {
  .drawer-hero,
  .timeline-top {
    grid-template-columns: 1fr;
    display: grid;
  }

  .drawer-meta,
  .drawer-details {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 520px) {
  .drawer-meta,
  .drawer-details {
    grid-template-columns: 1fr;
  }
}
</style>
