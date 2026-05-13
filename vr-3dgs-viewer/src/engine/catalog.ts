import type { BrainDanceViewerPayload } from '../types/viewer'
import { normalizePayload } from './payload'

const catalogUrl = import.meta.env.VITE_BD_MODEL_CATALOG_URL || './models/model_catalog.json'

export async function loadStandaloneCatalog(): Promise<BrainDanceViewerPayload | null> {
  try {
    const response = await fetch(catalogUrl, { cache: 'no-cache' })
    if (!response.ok) return null
    return normalizePayload(await response.json())
  } catch (error) {
    console.warn('[BrainDance VR] 独立模型目录加载失败:', error)
    return null
  }
}

export async function mergeStandaloneCatalog(seed: BrainDanceViewerPayload) {
  const catalog = await loadStandaloneCatalog()
  if (!catalog) return seed

  return {
    ...catalog,
    ...seed,
    modelList: seed.modelList?.length ? seed.modelList : catalog.modelList,
    markers: seed.markers?.length ? seed.markers : catalog.markers,
    searchResults: seed.searchResults?.length ? seed.searchResults : catalog.searchResults,
    authSession: seed.authSession || catalog.authSession,
    activeModelId: seed.activeModelId || catalog.activeModelId,
    sceneId: seed.sceneId || catalog.sceneId,
  } satisfies BrainDanceViewerPayload
}
