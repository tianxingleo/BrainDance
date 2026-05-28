import { viewerSupabaseClient } from '../engine/supabaseClient'
import { normalizeMatrixForViewer, normalizeVector3ForViewer } from '../engine/bridge'
import type {
  BrainDanceRecallMarker,
  BrainDanceRecallModel,
  BrainDanceRecallSearchResult,
} from '../types/viewer'

const defaultBucket = 'braindance-assets'
const signedUrlTtlSeconds = 60 * 60

type ModelAssetRow = {
  id: string
  scene_id: string | null
  display_name: string | null
  description: string | null
  tags: unknown
  ply_path: string | null
  preview_img_path: string | null
  meta_info: unknown
  created_at: string | null
}

type MemoryPoseRow = {
  id: string
  frame_index: number | null
  image_path: string | null
  image_name?: string | null
  pose_data: unknown
  caption: string | null
  tag?: string | null
  tags: unknown
  created_at: string | null
}

export type RemoteModelSource = 'mine' | 'community'

export function getViewerStorageBucket() {
  return import.meta.env.VITE_BD_SUPABASE_STORAGE_BUCKET?.trim() || defaultBucket
}

export function derivePosesPath(modelPath: string) {
  const cleanPath = modelPath.split('?')[0] || modelPath
  const derived = cleanPath.replace(
    /(?:point_cloud|model)\.(ply|splat|ksplat|spz)$/i,
    'webgl_poses.json',
  )
  return derived === cleanPath ? undefined : derived
}

export async function fetchRemoteModels(
  options: { source?: RemoteModelSource; query?: string; limit?: number } = {},
): Promise<BrainDanceRecallModel[]> {
  if (!viewerSupabaseClient) return []

  const source = options.source || 'mine'
  const limit = options.limit ?? 100
  let query = viewerSupabaseClient
    .from('model_assets')
    .select('id, scene_id, display_name, description, tags, ply_path, preview_img_path, meta_info, created_at')
    .not('ply_path', 'is', null)
    .order('created_at', { ascending: false })
    .limit(limit)

  const currentUserId = (await viewerSupabaseClient.auth.getUser()).data.user?.id
  if (source === 'mine' && currentUserId) {
    query = query.eq('user_id', currentUserId)
  }

  const keyword = options.query?.trim()
  if (keyword) {
    const escaped = escapePostgrestLike(keyword)
    query = query.or(`scene_id.ilike.%${escaped}%,display_name.ilike.%${escaped}%,description.ilike.%${escaped}%`)
  }

  const { data, error } = await query
  if (error) throw error

  const rows = (data ?? []) as ModelAssetRow[]
  const models = await Promise.all(rows.map(rowToModel))
  return models.filter((model): model is BrainDanceRecallModel => Boolean(model))
}

export async function fetchModelMarkers(modelId: string): Promise<BrainDanceRecallMarker[]> {
  if (!viewerSupabaseClient || !modelId || isLocalModelId(modelId)) return []

  const { data, error } = await viewerSupabaseClient
    .from('memory_poses')
    .select('id, frame_index, image_path, image_name, pose_data, caption, tag, tags, created_at')
    .eq('model_id', modelId)
    .order('frame_index', { ascending: true })
    .limit(200)

  if (error) throw error
  return ((data ?? []) as MemoryPoseRow[]).map(rowToMarker)
}

export function markersToSearchResults(markers: BrainDanceRecallMarker[]): BrainDanceRecallSearchResult[] {
  return markers.map((marker) => ({
    id: `pose-${marker.id}`,
    label: marker.label,
    description: marker.description,
    imageId: marker.imageId,
    matrix: marker.matrix,
    position: marker.position,
    markerId: marker.id,
    score: marker.score,
    tags: marker.tags,
    createdAt: marker.createdAt,
  }))
}

async function rowToModel(row: ModelAssetRow): Promise<BrainDanceRecallModel | null> {
  const modelPath = row.ply_path?.trim()
  if (!modelPath) return null

  const posesPath = getStringField(row.meta_info, ['poses_path', 'posesUrl', 'poses_url'])
    || derivePosesPath(modelPath)
  const modelUrl = await resolveStorageUrl(modelPath)
  const posesUrl = posesPath ? await resolveStorageUrl(posesPath) : undefined
  const previewImage = row.preview_img_path ? await resolveStorageUrl(row.preview_img_path) : undefined
  const displayName = row.display_name || row.scene_id || row.id

  return {
    id: String(row.id),
    sceneId: row.scene_id || undefined,
    name: displayName,
    displayName,
    ply: modelUrl,
    modelUrl,
    poses: posesUrl,
    posesUrl,
    previewImage,
    previewImg: previewImage,
    description: row.description || undefined,
    tags: normalizeTags(row.tags),
    createdAt: row.created_at || undefined,
  }
}

function rowToMarker(row: MemoryPoseRow): BrainDanceRecallMarker {
  const poseData = isRecord(row.pose_data) ? row.pose_data : {}
  const matrix = normalizeMatrixForViewer(
    poseData.matrix || poseData.transform_matrix || poseData.transform || poseData.camera_to_world,
  )
  const position = normalizeVector3ForViewer(
    poseData.position || poseData.translation || poseData.location,
  )
  const label = row.caption || row.image_name || row.image_path || `Frame ${row.frame_index ?? '-'}`
  const tags = normalizeTags(row.tags) || normalizeTags(row.tag)

  return {
    id: String(row.id),
    label,
    description: row.caption || undefined,
    imageId: row.image_path || row.image_name || undefined,
    matrix,
    position,
    color: '#9ed0c6',
    tags,
    createdAt: row.created_at || undefined,
  }
}

async function resolveStorageUrl(pathOrUrl: string): Promise<string> {
  const value = pathOrUrl.trim()
  if (!value || isAbsoluteUrl(value) || !viewerSupabaseClient) return value

  const { data, error } = await viewerSupabaseClient.storage
    .from(getViewerStorageBucket())
    .createSignedUrl(value, signedUrlTtlSeconds)

  if (error) throw error
  return data.signedUrl
}

function isAbsoluteUrl(value: string) {
  return /^https?:\/\//i.test(value) || value.startsWith('blob:')
}

function normalizeTags(value: unknown): string[] | undefined {
  if (Array.isArray(value)) {
    const tags = value.map(String).map((item) => item.trim()).filter(Boolean)
    return tags.length > 0 ? tags : undefined
  }
  if (typeof value === 'string' && value.trim()) {
    const tags = value.split(/[,，、\s]+/).map((item) => item.trim()).filter(Boolean)
    return tags.length > 0 ? tags : undefined
  }
  return undefined
}

function getStringField(value: unknown, keys: string[]) {
  if (!isRecord(value)) return undefined
  for (const key of keys) {
    const candidate = value[key]
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim()
  }
  return undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function escapePostgrestLike(value: string) {
  return value.replace(/[,%]/g, (match) => `\\${match}`)
}

function isLocalModelId(modelId: string) {
  return modelId === 'current'
    || modelId.startsWith('local-')
    || modelId.startsWith('blob:')
}
