import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import * as Dialog from '@radix-ui/react-dialog'
import { ArrowLeft, ArrowRight, Download, ImagePlus, Search, Sparkles, Star, ThumbsUp, ThumbsDown, Send, Grid3X3, Trash2, X } from 'lucide-react'
import { imageApi, type DetectedObject, type ImageHit, type ImageObjectFilter, type ImageSearchRequest, type DRESSession } from '../api/imageSearch'
import { Button, Spinner } from '../components/ui'
import { RetryImage } from '../components/RetryImage'
import { DRESAccess, DRESSubmit } from './DRESAccess'
import './imageSearch.css'

const emptyQuery: ImageSearchRequest = { text: '', objects: [], offset: 0, limit: 60 }
const shortlistKey = 'image-retrieval-shortlist-v1'
const historyKey = 'image-retrieval-history-v1'
function saved<T>(key: string, fallback: T): T {
  try { return JSON.parse(localStorage.getItem(key) ?? 'null') as T ?? fallback } catch { return fallback }
}
const hour = (h: number) => `${String(h).padStart(2, '0')}h`
const countOperator = (f: ImageObjectFilter) => f.max_count == null ? 'gte' : f.min_count === f.max_count ? 'eq' : f.min_count === 0 ? 'lte' : 'range'

export function ImageSearchPage() {
  useEffect(() => {
    document.documentElement.classList.add('image-monochrome')
    return () => document.documentElement.classList.remove('image-monochrome')
  }, [])
  return <DRESAccess>{(session, account) => <ImageSearchWorkbench session={session} onAccount={account} />}</DRESAccess>
}

function ImageSearchWorkbench({ session, onAccount }: { session: DRESSession; onAccount: () => void }) {
  const health = useQuery({ queryKey: ['image-health'], queryFn: imageApi.health, retry: 1, refetchInterval: 30_000 })
  const facets = useQuery({ queryKey: ['image-facets'], queryFn: imageApi.facets, retry: 1 })
  const [draft, setDraft] = useState<ImageSearchRequest>(emptyQuery)
  const [query, setQuery] = useState<ImageSearchRequest>(emptyQuery)
  const results = useQuery({ queryKey: ['image-results', query], queryFn: () => imageApi.search(query), placeholderData: previous => previous, retry: false })
  const [columns, setColumns] = useState(5)
  const [fit, setFit] = useState<'cover' | 'contain'>('contain')
  const [shortlist, setShortlist] = useState<string[]>(() => saved(shortlistKey, []))
  const [history, setHistory] = useState<ImageSearchRequest[]>(() => saved(historyKey, []))
  const [historyOpen, setHistoryOpen] = useState(false)
  const [onlySaved, setOnlySaved] = useState(false)
  const [selected, setSelected] = useState<string | null>(null)
  const [active, setActive] = useState(0)
  const [submitMessage, setSubmitMessage] = useState('')
  const quickSubmit = useMutation({ mutationFn: imageApi.dresSubmit, onSuccess: result => setSubmitMessage(`${result.submission ?? 'Đã gửi'}${result.description ? ` · ${result.description}` : ''}`), onError: error => setSubmitMessage(error.message) })
  const [preview, setPreview] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState('')
  const [agentInstruction, setAgentInstruction] = useState('')
  const [agentHistory, setAgentHistory] = useState<string[]>([])
  const [panel, setPanel] = useState<'search' | 'agent' | null>(null)
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; text: string; warnings?: string[] }[]>([])
  const chatEnd = useRef<HTMLDivElement>(null)
  const chatInput = useRef<HTMLTextAreaElement>(null)
  const composerRef = useRef<HTMLElement>(null)
  const draftRef = useRef(draft)
  useEffect(() => { draftRef.current = draft }, [draft])
  const agent = useMutation({ mutationFn: ({ instruction, current, history }: { instruction: string; current: ImageSearchRequest; history: string[] }) => imageApi.parse(instruction, current, history) })
  useEffect(() => { chatEnd.current?.scrollIntoView({ block: 'nearest' }) }, [messages, agent.isPending, panel])
  useEffect(() => { if (panel === 'agent') chatInput.current?.focus(); if (panel === 'search') composerRef.current?.querySelector('textarea')?.focus() }, [panel])
  const fileInput = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLElement>(null)
  const encoderAvailable = !!health.data && health.data.queryEncoder !== 'disabled'
  const upload = useMutation({ mutationFn: imageApi.upload })

  useEffect(() => { localStorage.setItem(shortlistKey, JSON.stringify(shortlist)) }, [shortlist])
  useEffect(() => { if (onlySaved) setQuery(q => ({ ...q, shortlist_ids: shortlist, offset: 0 })) }, [shortlist, onlySaved])
  useEffect(() => { localStorage.setItem(historyKey, JSON.stringify(history)) }, [history])
  useEffect(() => { setActive(0) }, [query])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  const toggleSaved = (id: string) => setShortlist(ids => ids.includes(id) ? ids.filter(i => i !== id) : [...ids, id])
  const runSearch = (next = draft, savedOnly = onlySaved) => {
    const applied = { ...next, offset: 0, shortlist_ids: savedOnly ? shortlist : undefined }
    setDraft(next); setQuery(applied)
    if (window.innerWidth <= 740 && panel !== 'agent') resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    // Uploaded assets belong to this backend session and are not persisted in history.
    if (!next.query_asset) setHistory(items => [{ ...next, offset: 0 }, ...items].slice(0, 20))
  }
  const similar = (id: string) => {
    const next = { ...draft, reference_id: id, query_asset: undefined, text: '', positive_ids: [], negative_ids: [] }
    setPreview(null); setSelected(null); runSearch(next)
  }
  const feedback = (id: string, kind: 'positive_ids' | 'negative_ids') => {
    const other = kind === 'positive_ids' ? 'negative_ids' : 'positive_ids'
    const values = query[kind] ?? []
    const patch = { [kind]: values.includes(id) ? values.filter(i => i !== id) : [...values, id].slice(-100), [other]: (query[other] ?? []).filter(i => i !== id), offset: 0 }
    setQuery(q => ({ ...q, ...patch }))
    setDraft(d => ({ ...d, ...patch }))
  }
  const resetFeedback = () => {
    const patch = { positive_ids: [], negative_ids: [], offset: 0 }
    setQuery(q => ({ ...q, ...patch })); setDraft(d => ({ ...d, ...patch }))
  }
  const submitImage = (imageID: string) => {
    if (quickSubmit.isPending || !session.connected || !session.evaluation_id) return
    setSubmitMessage('')
    quickSubmit.mutate({ image_id: imageID, media_item_name: imageID, collection: '', task_id: '' })
  }
  const acceptImage = async (file?: File) => {
    if (!file) return
    setUploadError('')
    if (!file.type.startsWith('image/')) { setUploadError('Hãy chọn một file ảnh.'); return }
    try {
      const asset = await upload.mutateAsync(file)
      setPreview(URL.createObjectURL(file))
      setDraft(d => ({ ...d, query_asset: asset.asset, reference_id: undefined }))
    } catch (error) { setUploadError((error as Error).message) }
  }
  const clearImage = () => { setPreview(null); setDraft(d => ({ ...d, query_asset: undefined, reference_id: undefined })) }
  const items = results.data?.items
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey && ['s','a'].includes(event.key.toLowerCase()) && !selected && !historyOpen) {
        event.preventDefault()
        if (!event.repeat) { const next = event.key.toLowerCase() === 's' ? 'search' : 'agent'; setPanel(p => p === next ? null : next) }
        return
      }
      if (selected || historyOpen || event.defaultPrevented || event.isComposing || event.altKey || event.ctrlKey || event.metaKey || (event.target instanceof HTMLElement && event.target.closest('input,textarea,select,button,a,[contenteditable="true"]'))) return
      if (!items?.length) return
      const delta: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: columns, ArrowUp: -columns }
      if (event.key in delta) { event.preventDefault(); setActive(i => Math.max(0, Math.min(items.length-1, i+delta[event.key]))) }
      if (event.key === 'Enter') { event.preventDefault(); setSelected(items[active].imageID) }
      if (event.key.toLowerCase() === 's') { event.preventDefault(); toggleSaved(items[active].imageID) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [items, active, columns, selected, historyOpen])

  const addFilter = () => setDraft(d => ({ ...d, objects: [...d.objects, { label: facets.data?.labels[0] ?? 'person', min_count: 1 }] }))
  const updateFilter = (i: number, patch: Partial<ImageObjectFilter>) => setDraft(d => ({ ...d, objects: d.objects.map((f, j) => j === i ? { ...f, ...patch } : f) }))
  const changeOperator = (i: number, op: string) => {
    const f = draft.objects[i]
    const value = f.min_count === 0 ? f.max_count ?? 1 : f.min_count
    updateFilter(i, op === 'eq' ? { min_count: value, max_count: value } : op === 'lte' ? { min_count: 0, max_count: value } : op === 'range' ? { min_count: value, max_count: Math.max(value, f.max_count ?? value+1) } : { min_count: value, max_count: null })
  }
  const askAgent = async () => {
    if (!agentInstruction.trim() || agent.isPending) return
    const instruction = agentInstruction.trim()
    const snapshot = JSON.stringify(draft)
    setMessages(m => [...m, { role: 'user', text: instruction }])
    setAgentInstruction('')
    try {
      const answer = await agent.mutateAsync({ instruction, current: draft, history: agentHistory })
      setAgentHistory(h => [...h, `User: ${instruction}`, `Agent: ${answer.message}`].slice(-6))
      if (snapshot !== JSON.stringify(draftRef.current)) {
        setMessages(m => [...m, { role: 'assistant', text: 'Điều kiện đã thay đổi trong lúc agent chạy. Gửi lại yêu cầu để sửa trên điều kiện mới nhất.' }]); return
      }
      setMessages(m => [...m, { role: 'assistant', warnings: answer.warnings, text: `${answer.message} · ${answer.engine === 'local' ? 'Cục bộ · ' : answer.engine === 'cache' ? 'Cache · ' : 'Luna · '}${(answer.elapsedMs/1000).toFixed(1)}s${answer.usage?.input_tokens !== undefined ? ` · ${answer.usage.input_tokens} input / ${answer.usage.output_tokens ?? 0} output token` : ''}` }])
      if (!answer.needs_clarification) runSearch(answer.query)
    } catch (error) { setMessages(m => [...m, { role: 'assistant', text: (error as Error).message }]) }
  }
  const exportSaved = () => {
    const blob = new Blob([['imageID', ...shortlist].join('\n')], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'image-shortlist.csv'; a.click(); URL.revokeObjectURL(url)
  }

  return <div className="app-shell image-app">
    <header className="app-header">
      <a href="/images" className="brand"><span className="brand-mark"><Sparkles size={16} /></span>IMAGE RETRIEVAL</a>
      <span className="muted small">{health.data ? `${health.data.images.toLocaleString()} ảnh · ${health.data.objects.toLocaleString()} objects` : 'Kết nối dữ liệu…'}</span>
      <span className="header-spacer" />
      <Button className={panel === 'search' ? 'primary sm' : 'sm'} aria-expanded={panel === 'search'} aria-controls="manual-search-panel" onClick={() => setPanel(p => p === 'search' ? null : 'search')}><Search size={14} />Tìm kiếm <kbd>Ctrl S</kbd></Button>
      <Button className={panel === 'agent' ? 'primary sm' : 'sm'} aria-expanded={panel === 'agent'} aria-controls="agent-chat-panel" onClick={() => setPanel(p => p === 'agent' ? null : 'agent')}><Sparkles size={14} />Agent <kbd>Ctrl A</kbd></Button>
      <Button className="sm" onClick={onAccount}>{session.connected ? `DRES · ${session.username}` : 'DRES'}</Button>
      <Button className={onlySaved ? 'primary sm' : 'sm'} onClick={() => { setOnlySaved(v => !v); runSearch(draft, !onlySaved) }}><Star size={14} />Shortlist ({shortlist.length})</Button>
      <Button className="sm" onClick={() => setHistoryOpen(true)}>Lịch sử</Button>
      <Button className="sm" disabled={!shortlist.length} onClick={exportSaved}><Download size={14} />CSV</Button>
      <span className="api-status"><span className={`status-dot ${health.isSuccess ? 'online' : 'offline'}`} />{health.isSuccess ? 'DATA READY' : 'API OFFLINE'}</span>
    </header>
    <main className={`main-content image-workbench ${panel ? `has-${panel}` : ''}`}>
      {panel === 'search' && <aside ref={composerRef} id="manual-search-panel" aria-label="Tìm kiếm thủ công" className="image-composer panel">
        <div className="row"><h2 className="grow">Tìm ảnh</h2><Button className="icon sm" aria-label="Thu gọn tìm kiếm" onClick={() => setPanel(null)}><X size={15} /></Button></div><span className="small muted">Qwen3-VL · 8B · Ctrl+S bật/tắt</span>
        {health.isError && <p role="alert" className="image-error">Không kết nối được backend ảnh. Chạy search_api.py ở cổng 8100.</p>}
        <form className="stack" onSubmit={event => { event.preventDefault(); runSearch() }}>
          <label className="field">Mô tả nội dung<textarea rows={4} value={draft.text} onChange={e => setDraft(d => ({ ...d, text: e.target.value }))} placeholder="Ví dụ: người áo đỏ đứng cạnh bồn rửa…" disabled={!encoderAvailable} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runSearch() } }} /></label>
          {!encoderAvailable && <p className="small muted">Chưa kết nối query encoder. Lọc dữ liệu và “Tìm tương tự” bằng ảnh trong tập đã dùng được.</p>}
          <div className="image-drop" tabIndex={0} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); void acceptImage(e.dataTransfer.files[0]) }} onPaste={e => { const file = Array.from(e.clipboardData.files).find(f => f.type.startsWith('image/')); if (file) { e.preventDefault(); void acceptImage(file) } }}>
            {preview ? <img src={preview} alt="Ảnh query" /> : draft.reference_id ? <img src={`/api/images/media/${encodeURIComponent(draft.reference_id)}?thumbnail=true`} alt="Ảnh tham chiếu" /> : <button type="button" className="image-drop-label" disabled={!encoderAvailable || upload.isPending} onClick={() => fileInput.current?.click()}><ImagePlus size={23} /><strong>Ảnh query</strong><span className="muted">Upload · kéo thả · Ctrl+V</span></button>}
            <input ref={fileInput} type="file" accept="image/*" hidden onChange={e => void acceptImage(e.target.files?.[0])} />
          </div>
          {(draft.reference_id || draft.query_asset) && <div className="row"><span className="small muted grow">{draft.reference_id ?? 'Ảnh upload'}</span><Button type="button" className="icon sm" aria-label="Bỏ ảnh query" onClick={clearImage}><Trash2 size={14} /></Button></div>}
          {upload.isPending && <Spinner />}{uploadError && <p role="alert" className="image-error">{uploadError}</p>}
          <div className="image-filter-pair">
            <label className="field">Ngày<select value={draft.day ?? ''} onChange={e => setDraft(d => ({ ...d, day: e.target.value || undefined }))}><option value="">Tất cả</option>{facets.data?.days.map(x => <option key={x}>{x}</option>)}</select></label>
            <label className="field">Nguồn<select value={draft.source ?? ''} onChange={e => setDraft(d => ({ ...d, source: e.target.value || undefined }))}><option value="">Tất cả</option>{facets.data?.sources.map(x => <option key={x}>{x}</option>)}</select></label>
            <label className="field">Từ giờ<select value={draft.hour_from ?? ''} onChange={e => setDraft(d => ({ ...d, hour_from: e.target.value === '' ? undefined : Number(e.target.value) }))}><option value="">Không giới hạn</option>{Array.from({length:24}, (_, x) => <option key={x} value={x}>{hour(x)}</option>)}</select></label>
            <label className="field">Đến giờ<select value={draft.hour_to ?? ''} onChange={e => setDraft(d => ({ ...d, hour_to: e.target.value === '' ? undefined : Number(e.target.value) }))}><option value="">Không giới hạn</option>{Array.from({length:24}, (_, x) => <option key={x} value={x}>{hour(x)}</option>)}</select></label>
          </div>
          <div className="row"><strong className="grow">Object / color</strong><Button type="button" className="sm" onClick={addFilter} disabled={!facets.data || draft.objects.length >= 12}>+ Điều kiện</Button></div>

          {draft.objects.map((f, i) => <div className="image-object-filter" key={i}>
            <div className="image-object-filter-main">
            <select aria-label={`Object ${i+1}`} value={f.label} onChange={e => updateFilter(i, { label: e.target.value })}>{facets.data?.labels.map(x => <option key={x}>{x}</option>)}</select>
            <select aria-label={`Màu ${i+1}`} disabled={f.count_source === 'metadata'} value={f.color ?? ''} onChange={e => updateFilter(i, { color: e.target.value || undefined })}><option value="">Mọi màu</option>{facets.data?.colors.map(x => <option key={x}>{x}</option>)}</select>
            <Button type="button" className="icon sm" aria-label={`Bỏ điều kiện ${i+1}`} onClick={() => setDraft(d => ({ ...d, objects: d.objects.filter((_, j) => j !== i) }))}><X size={13} /></Button>
            </div><div className="image-object-filter-counts">
            <select aria-label={`So sánh số lượng ${i+1}`} value={countOperator(f)} onChange={e => changeOperator(i, e.target.value)}><option value="gte">Ít nhất ≥</option><option value="eq">Đúng =</option><option value="lte">Tối đa ≤</option><option value="range">Khoảng</option></select>
            <input aria-label={`Số lượng ${i+1}`} type="number" min={0} max={100} value={countOperator(f) === 'lte' ? f.max_count ?? 0 : f.min_count} onChange={e => { const n = Math.max(0, Math.min(100, Number(e.target.value))); updateFilter(i, countOperator(f) === 'eq' ? { min_count: n, max_count: n } : countOperator(f) === 'lte' ? { min_count: 0, max_count: n } : { min_count: n }) }} />
            {countOperator(f) === 'range' && <input aria-label={`Số lượng tối đa ${i+1}`} type="number" min={f.min_count} max={100} value={f.max_count ?? f.min_count} onChange={e => updateFilter(i, { max_count: Math.max(f.min_count, Math.min(100, Number(e.target.value))) })} />}
            </div><select aria-label={`Nguồn số đếm ${i+1}`} value={f.count_source ?? 'detected'} onChange={e => updateFilter(i, { count_source: e.target.value as 'detected' | 'metadata' })}><option value="detected">Detection mới · có màu</option><option value="metadata" disabled={!!f.color}>Metadata gốc · không có màu</option></select>
          </div>)}
          <label className="row small"><input type="checkbox" checked={draft.rerank ?? false} disabled={!health.data?.queryReranker || health.data.queryReranker === 'disabled'} onChange={e => setDraft(d => ({ ...d, rerank: e.target.checked }))} />Qwen reranker</label>
          <Button className="primary" type="submit" disabled={upload.isPending || (!!draft.query_asset && !encoderAvailable)}>{results.isFetching ? <Spinner /> : <Search size={16} />}Tìm ảnh</Button>
          <Button type="button" onClick={() => { clearImage(); setOnlySaved(false); runSearch(emptyQuery, false) }}>Xóa điều kiện</Button>
        </form>
      </aside>}
      <section className="image-results" ref={resultsRef}>
        <div className="row image-results-toolbar"><div className="grow"><h1>{onlySaved ? 'Shortlist' : 'Khám phá ảnh'}</h1><span className="small muted">{results.data ? `${results.data.total.toLocaleString()} kết quả · ${results.data.reranked ? 'Qwen reranker' : results.data.ranked ? 'cosine similarity' : 'thứ tự ngày / giờ / frame'}` : 'Đang tải…'}</span></div>
          <label className="row small">Cột <input aria-label="Số cột" type="range" min={3} max={8} value={columns} onChange={e => setColumns(Number(e.target.value))} /></label>
          <Button className="sm" onClick={() => setFit(f => f === 'contain' ? 'cover' : 'contain')}>{fit}</Button>
        </div>
        {((query.positive_ids?.length ?? 0) + (query.negative_ids?.length ?? 0) > 0) && <div className="row image-feedback-summary"><ThumbsUp size={13} />{query.positive_ids?.length ?? 0}<ThumbsDown size={13} />{query.negative_ids?.length ?? 0}<Button className="sm" onClick={resetFeedback}>Bỏ feedback</Button>{results.isFetching && <Spinner />}</div>}
        {submitMessage && <div className="row image-submit-status" role="status"><span className="grow">{submitMessage}</span><Button className="icon sm" aria-label="Ẩn trạng thái nộp bài" onClick={() => setSubmitMessage('')}><X size={13} /></Button></div>}
        {results.isError && <div className="empty-state image-error" role="alert">{results.error.message}<Button onClick={() => void results.refetch()}>Thử lại</Button></div>}
        {results.isPending && <div className="empty-state"><Spinner />Đang tải ảnh…</div>}
        {results.data?.total === 0 && <div className="empty-state">Không có ảnh phù hợp. Thử giảm số điều kiện hoặc bỏ màu.</div>}
        <div className="keyframe-grid" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
          {items?.map((hit, i) => <article className={`keyframe-tile ${active === i ? 'active' : ''}`} key={hit.imageID} tabIndex={0} onFocus={() => setActive(i)} onClick={() => setSelected(hit.imageID)} onDoubleClick={() => similar(hit.imageID)}>
            <RetryImage className={fit} src={hit.thumbnailUrl} alt={hit.imageID} loading="lazy" />
            <span className="rank-chip">#{query.offset+i+1}</span>
            <button className={`tile-star ${shortlist.includes(hit.imageID) ? 'active' : ''}`} aria-label={`Shortlist ${hit.imageID}`} onClick={e => { e.stopPropagation(); toggleSaved(hit.imageID) }}><Star size={15} fill={shortlist.includes(hit.imageID) ? 'currentColor' : 'none'} /></button>
            <div className="image-tile-actions" onClick={e => e.stopPropagation()} onDoubleClick={e => e.stopPropagation()}>
              <button className={query.positive_ids?.includes(hit.imageID) ? 'liked' : ''} aria-label={`Like ${hit.imageID}`} aria-pressed={query.positive_ids?.includes(hit.imageID) ?? false} title="Like" onClick={() => feedback(hit.imageID, 'positive_ids')}><ThumbsUp size={15} /></button>
              <button className={query.negative_ids?.includes(hit.imageID) ? 'disliked' : ''} aria-label={`Dislike ${hit.imageID}`} aria-pressed={query.negative_ids?.includes(hit.imageID) ?? false} title="Dislike" onClick={() => feedback(hit.imageID, 'negative_ids')}><ThumbsDown size={15} /></button>
              <button aria-label={`Submit ${hit.imageID}`} title={session.connected ? 'Submit' : 'Đăng nhập DRES'} disabled={!session.connected || !session.evaluation_id || quickSubmit.isPending} onClick={() => submitImage(hit.imageID)}>{quickSubmit.isPending && quickSubmit.variables?.image_id === hit.imageID ? <Spinner /> : <Send size={15} />}</button>
            </div>
            <footer className="tile-overlay"><span>{hit.source} · {hour(hit.hour)}</span><span>f{hit.frame}</span>{hit.score !== null && <span className="score-chip">{(hit.rerankScore ?? hit.score).toFixed(3)}</span>}</footer>
            {!!hit.constraintCounts?.length && <div className="image-count-evidence">{hit.constraintCounts.slice(0,2).map((f,j) => <span key={j}>{f.label}{f.color ? `/${f.color}` : ''}: {f.count} {f.source === 'metadata' ? '(gốc)' : ''}</span>)}</div>}
          </article>)}
        </div>
        <div className="row image-pagination"><span className="small muted grow">Mũi tên: chọn ảnh · Enter: inspect · S: shortlist</span><Button disabled={query.offset === 0 || results.isFetching} onClick={() => setQuery(q => ({ ...q, offset: Math.max(0, q.offset-q.limit) }))}><ArrowLeft size={14} /></Button><span className="small">{Math.floor(query.offset/query.limit)+1} / {Math.max(1, Math.ceil((results.data?.total ?? 0)/query.limit))}</span><Button disabled={query.offset+query.limit >= (results.data?.total ?? 0) || results.isFetching} onClick={() => setQuery(q => ({ ...q, offset: q.offset+q.limit }))}><ArrowRight size={14} /></Button></div>
      </section>
      {panel === 'agent' && <aside id="agent-chat-panel" aria-label="Chat agent tìm kiếm" className="image-chat panel">
        <div className="row"><div className="grow"><strong>Trợ lý tìm ảnh</strong><div className="small muted">GPT-6 Luna · Ctrl+A bật/tắt</div></div><Button className="icon sm" aria-label="Thu gọn agent" onClick={() => setPanel(null)}><X size={15} /></Button></div>
        <div className="image-chat-log" role="log" aria-label="Hội thoại tìm kiếm" aria-live="polite">
          {messages.map((message, i) => <div key={i} className={`image-chat-bubble ${message.role}`}><span className="small muted">{message.role === 'user' ? 'Bạn' : 'Trợ lý'}</span><p>{message.text}</p>{message.warnings?.map((w,j) => <p className="image-agent-warning small" key={j}>{w}</p>)}</div>)}
          {agent.isPending && <div className="image-chat-bubble assistant" role="status"><Spinner /> Đang hiểu yêu cầu…</div>}<div ref={chatEnd} />
        </div>
        <form className="image-chat-compose stack" onSubmit={e => { e.preventDefault(); void askAgent() }}>
          <textarea ref={chatInput} aria-label="Yêu cầu agent" rows={3} value={agentInstruction} onChange={e => setAgentInstruction(e.target.value)} placeholder="Mô tả ảnh hoặc chỉnh điều kiện…" onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void askAgent() } }} />
          <div className="row"><Button type="button" className="sm" disabled={agent.isPending || !messages.length} onClick={() => { setMessages([]); setAgentHistory([]) }}>Xóa chat</Button><span className="grow" /><Button type="submit" className="primary" disabled={agent.isPending || !agentInstruction.trim()}>{agent.isPending ? <Spinner /> : <Sparkles size={14} />}Hiểu & tìm</Button></div>
          <span className="small muted">Enter gửi · Shift+Enter xuống dòng</span>
        </form>
      </aside>}
    </main>
    <ImageInspector session={session} id={selected} onClose={() => setSelected(null)} onSelect={setSelected} onSimilar={similar} saved={!!selected && shortlist.includes(selected)} onSave={() => selected && toggleSaved(selected)} />
    <Dialog.Root open={historyOpen} onOpenChange={setHistoryOpen}><Dialog.Portal><Dialog.Overlay className="dialog-overlay" /><Dialog.Content className="dialog-content"><Dialog.Title>Lịch sử tìm ảnh</Dialog.Title><Dialog.Description className="muted small">Lưu trên trình duyệt này. Ảnh upload không được lưu lại.</Dialog.Description><div className="stack">{history.length === 0 && <p>Chưa có truy vấn.</p>}{history.map((q, i) => <Button key={i} onClick={() => { setPreview(null); setOnlySaved(false); runSearch(q, false); setHistoryOpen(false) }}>{q.text || q.reference_id || `${q.objects.length} điều kiện object · ${q.day ?? 'mọi ngày'} · ${q.hour_from ?? '…'}–${q.hour_to ?? '…'}h`}</Button>)}</div><Dialog.Close asChild><Button>Đóng</Button></Dialog.Close></Dialog.Content></Dialog.Portal></Dialog.Root>
  </div>
}

function ImageInspector({ session, id, onClose, onSelect, onSimilar, saved, onSave }: { session: DRESSession; id: string | null; onClose: () => void; onSelect: (id: string) => void; onSimilar: (id: string) => void; saved: boolean; onSave: () => void }) {
  const detail = useQuery({ queryKey: ['image-detail', id], queryFn: () => imageApi.detail(id!), enabled: !!id })
  const [object, setObject] = useState<DetectedObject | null>(null)
  const [maskOn, setMaskOn] = useState(false)
  const [timeline, setTimeline] = useState(false)
  const inspector = useRef<HTMLDivElement>(null)
  const data = detail.data
  const video = useQuery({ queryKey: ['image-video', data?.videoID], queryFn: () => imageApi.video(id!), enabled: !!data && !!id && timeline })
  useEffect(() => { if (timeline) inspector.current?.querySelector('.image-video-grid button.active')?.scrollIntoView({ block: 'center' }) }, [timeline, video.data, id])
  useEffect(() => { setObject(null); setMaskOn(false) }, [id])
  useEffect(() => { if (!id) setTimeline(false) }, [id])
  useEffect(() => {
    if (!id || !data) return
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.isComposing) return
      const target = event.target instanceof HTMLElement ? event.target : null
      if (target?.closest('input,textarea,select,[contenteditable="true"]')) return
      const dialog = target?.closest('[role="dialog"]')
      if (dialog && dialog !== inspector.current) return
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
      event.preventDefault(); event.stopPropagation()
      const next = event.key === 'ArrowLeft' ? data.previousImageID : data.nextImageID
      if (next) onSelect(next)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [id, data, onSelect])
  const inspect = (imageID: string) => { setTimeline(false); onSelect(imageID) }
  return <Dialog.Root open={!!id} onOpenChange={open => { if (!open) onClose() }}><Dialog.Portal><Dialog.Overlay className="dialog-overlay" /><Dialog.Content ref={inspector} className={`dialog-content image-inspector ${timeline ? 'is-video-grid' : ''}`}>
    <div className="row"><Dialog.Title className="grow">{timeline ? data?.videoID ?? id : id}</Dialog.Title><Dialog.Close asChild><Button className="icon" aria-label="Đóng inspector"><X size={17} /></Button></Dialog.Close></div>
    <Dialog.Description className="sr-only">Ảnh và các frame cùng video.</Dialog.Description>
    {detail.isPending && <Spinner />}{detail.isError && <p role="alert">{detail.error.message}</p>}
    {data && <>
      <div className="row wrap image-video-toolbar">
        <Button aria-label="Frame trước" title="←" disabled={!data.previousImageID} onClick={() => data.previousImageID && onSelect(data.previousImageID)}><ArrowLeft size={16} /></Button>
        <span className="grow small muted">{data.videoID} · {data.videoFramePosition+1}/{data.videoFrameCount} · f{data.frame}</span>
        <Button className={timeline ? 'primary sm' : 'sm'} onClick={() => setTimeline(v => !v)}><Grid3X3 size={15} />{timeline ? 'Xem ảnh' : 'Tất cả frame'}</Button>
        <Button aria-label="Frame sau" title="→" disabled={!data.nextImageID} onClick={() => data.nextImageID && onSelect(data.nextImageID)}><ArrowRight size={16} /></Button>
      </div>
      {timeline ? <div className="image-video-grid" aria-label="Tất cả frame trong video">
        {video.isPending && <Spinner />}{video.isError && <p role="alert">{video.error.message}</p>}
        {video.data?.items.map(hit => <button key={hit.imageID} aria-label={`Xem frame ${hit.frame}`} className={hit.imageID === id ? 'active' : ''} onClick={() => inspect(hit.imageID)}><img src={hit.thumbnailUrl} alt={hit.imageID} loading="lazy" /><span>f{hit.frame}</span></button>)}
      </div> : <><div className="image-inspector-body"><div className="image-inspector-picture"><img src={data.imageUrl} alt={data.imageID} />{object && maskOn && <img className="image-mask-layer" src={`/api/images/mask/${encodeURIComponent(object.instance_id)}`} alt={`Mask ${object.label}`} />}</div><aside className="stack image-inspector-meta">
        <div className="row wrap"><Button className="primary" onClick={() => onSimilar(data.imageID)}><Search size={14} />Tìm tương tự</Button><Button onClick={onSave}><Star size={14} fill={saved ? 'currentColor' : 'none'} />{saved ? 'Đã lưu' : 'Lưu'}</Button><DRESSubmit key={data.imageID} imageID={data.imageID} session={session} /></div>
        <strong>{data.objects.length} objects</strong>
        <div className="image-object-list">{data.objects.map(obj => <button key={obj.instance_id} className={`image-object-row ${object?.instance_id === obj.instance_id ? 'active' : ''}`} onClick={() => { setObject(obj); setMaskOn(true) }}><strong>{obj.label}</strong><span>{(obj.confidence*100).toFixed(0)}%</span><span>{Object.entries(obj.colors).filter(([, fraction]) => fraction >= .1).map(([name, fraction]) => `${name} ${(fraction*100).toFixed(0)}%`).join(' · ')}</span></button>)}</div>
        {object && <Button className="sm" onClick={() => setMaskOn(v => !v)}>{maskOn ? 'Ẩn' : 'Hiện'} mask {object.label}</Button>}
        <details><summary>Số đếm</summary><table className="image-count-table"><thead><tr><th>Object</th><th>Detection</th><th>Metadata</th></tr></thead><tbody>{Array.from(new Set([...Object.keys(data.objectCounts), ...Object.keys(data.metadataObjectCounts ?? {})])).sort().map(label => <tr key={label}><td>{label}</td><td>{data.objectCounts[label] ?? 0}</td><td>{data.metadataObjectCounts?.[label] ?? '—'}</td></tr>)}</tbody></table></details>
        <details><summary>Mô tả</summary><p className="small muted">{data.originalDescription}</p></details>
      </aside></div><div className="image-neighbors">{data.neighbors.map((hit: ImageHit) => <button key={hit.imageID} className={hit.imageID === id ? 'active' : ''} onClick={() => inspect(hit.imageID)}><img src={hit.thumbnailUrl} alt={hit.imageID} /><span>f{hit.frame}</span></button>)}</div></>}
    </>}
  </Dialog.Content></Dialog.Portal></Dialog.Root>
}
