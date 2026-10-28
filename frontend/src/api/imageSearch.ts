export type ImageHit = {
  imageID: string; day: string; hour: number; frame: number; source: string; batch: string; videoID: string
  score: number | null; rerankScore?: number; thumbnailUrl: string; imageUrl: string; objectCounts: Record<string, number>
  metadataObjectCounts: Record<string, number>
  constraintCounts?: { label: string; color: string | null; source: 'detected' | 'metadata'; count: number }[]
}
export type DetectedObject = {
  instance_id: string; label: string; confidence: number; bbox_xyxy: number[]
  mask_quality: number; mask_area: number; dominant_color: string; colors: Record<string, number>
}
export type ImageDetail = ImageHit & { objects: DetectedObject[]; neighbors: ImageHit[]; originalDescription: string; videoFrameCount: number; videoFramePosition: number; previousImageID: string | null; nextImageID: string | null }
export type DRESSession = { connected: boolean; server?: string; username?: string; evaluation_id?: string }
export type DRESRun = { id: string; name: string; status: string }
export type ImageObjectFilter = { label: string; color?: string | null; min_count: number; max_count?: number | null; count_source?: 'detected' | 'metadata'; min_color_fraction?: number }
export type ImageSearchRequest = {
  text: string; reference_id?: string; query_asset?: string; day?: string; source?: string
  hour_from?: number; hour_to?: number; objects: ImageObjectFilter[]; shortlist_ids?: string[]
  rerank?: boolean; rerank_k?: number
  positive_ids?: string[]; negative_ids?: string[]
  offset: number; limit: number
}
export type ImageResults = { items: ImageHit[]; total: number; ranked: boolean; reranked?: boolean; offset: number; limit: number }
export type ImageFacets = { days: string[]; sources: string[]; labels: string[]; colors: string[]; hours: number[] }
export type ImageHealth = { status: string; images: number; objects: number; queryEncoder: string; queryReranker?: string; model: string }
export type AgentAnswer = { model: string; engine?: 'local' | 'cache' | 'luna'; usage?: { input_tokens?: number; output_tokens?: number }; message: string; warnings: string[]; needs_clarification: boolean; query: ImageSearchRequest; elapsedMs: number }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/images${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: unknown }
    const message = typeof body.detail === 'string' ? body.detail : Array.isArray(body.detail)
      ? body.detail.map((item: { msg?: string }) => item.msg ?? 'Tham số không hợp lệ').join('; ')
      : `API ảnh trả lỗi ${response.status}. Kiểm tra backend và tham số tìm.`
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const imageApi = {
  dresSession: () => request<DRESSession>('/dres/session'),
  dresLogin: (body: { server: string; username: string; password: string; session_id: string }) => request<DRESSession & { evaluations: DRESRun[] }>('/dres/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  dresRuns: () => request<DRESRun[]>('/dres/evaluations'),
  dresSelect: (evaluation_id: string) => request<DRESSession>('/dres/evaluation', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ evaluation_id }) }),
  dresLogout: () => request<DRESSession>('/dres/logout', { method: 'POST' }),
  dresSubmit: (body: { image_id: string; media_item_name: string; collection: string; task_id: string }) => request<{ http_status: number; status: boolean; submission?: string; description: string }>('/dres/submit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  video: (id: string) => request<{ videoID: string; items: ImageHit[] }>(`/video/${encodeURIComponent(id)}`),
  health: () => request<ImageHealth>('/health'),
  facets: () => request<ImageFacets>('/facets'),
  search: (query: ImageSearchRequest) => request<ImageResults>('/search', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(query),
  }),
  detail: (id: string) => request<ImageDetail>(`/detail/${encodeURIComponent(id)}`),
  parse: (instruction: string, current: ImageSearchRequest, history: string[]) => request<AgentAnswer>('/agent/parse', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ instruction, current, history }),
  }),
  upload: (file: File) => {
    const body = new FormData(); body.append('image', file)
    return request<{ asset: string }>('/query-assets', { method: 'POST', body })
  },
}
