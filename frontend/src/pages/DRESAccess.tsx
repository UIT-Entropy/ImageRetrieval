import { useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import * as Dialog from '@radix-ui/react-dialog'
import { LogIn, Send, Sparkles } from 'lucide-react'
import { imageApi, type DRESSession } from '../api/imageSearch'
import { Button, Spinner } from '../components/ui'

export function DRESAccess({ children }: { children: (session: DRESSession, account: () => void) => ReactNode }) {
  const cache = useQueryClient()
  const session = useQuery({ queryKey: ['dres-session'], queryFn: imageApi.dresSession, retry: false })
  const [practice, setPractice] = useState(() => sessionStorage.getItem('image-practice') === 'true')
  const [access, setAccess] = useState(false)
  const [server, setServer] = useState(() => localStorage.getItem('image-dres-server') ?? '')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [token, setToken] = useState('')
  const [mode, setMode] = useState('password')
  const [evaluation, setEvaluation] = useState('')
  const login = useMutation({ mutationFn: imageApi.dresLogin, onSuccess: value => {
    cache.setQueryData(['dres-session'], value); cache.setQueryData(['dres-runs'], value.evaluations)
    localStorage.setItem('image-dres-server', server.trim()); setPassword(''); setToken('')
  } })
  const runs = useQuery({ queryKey: ['dres-runs'], queryFn: imageApi.dresRuns, enabled: !!session.data?.connected && (access || !session.data?.evaluation_id), retry: false })
  const select = useMutation({ mutationFn: imageApi.dresSelect, onSuccess: value => {
    cache.setQueryData(['dres-session'], value); setAccess(false); setPractice(false); sessionStorage.removeItem('image-practice')
  } })
  const logout = useMutation({ mutationFn: imageApi.dresLogout, onSuccess: value => {
    cache.setQueryData(['dres-session'], value); cache.removeQueries({ queryKey: ['dres-runs'] }); setEvaluation(''); setPractice(false); sessionStorage.removeItem('image-practice')
  } })
  if (session.isPending) return <div className="dres-access"><Spinner /></div>
  const connected = !!session.data?.connected
  if (!access && (practice || (connected && session.data?.evaluation_id))) return children(practice ? { connected: false } : session.data ?? { connected: false }, () => setAccess(true))
  return <main className="dres-access"><section className="panel dres-login stack">
    <span className="dres-brand"><Sparkles size={24} /> IMAGE RETRIEVAL</span>
    <h1>{connected ? 'Chọn lượt thi' : 'Đăng nhập DRES'}</h1>
    {session.isError && <p role="alert" className="image-error">{session.error.message}</p>}
    {connected ? <><p>{session.data?.username} · {session.data?.server}</p>
      <label className="field">Evaluation ID<select aria-label="Lượt thi DRES" value={evaluation} onChange={e => setEvaluation(e.target.value)}><option value="">Chọn lượt thi…</option>{(runs.data ?? login.data?.evaluations ?? []).map(run => <option key={run.id} value={run.id}>{run.name} · {run.status} · {run.id}</option>)}</select></label>
      <div className="row wrap"><Button disabled={runs.isFetching} onClick={() => void runs.refetch()}>Làm mới lượt thi</Button><Button disabled={logout.isPending} onClick={() => logout.mutate()}>Đăng xuất</Button></div>
      <Button className="primary" disabled={!evaluation || select.isPending} onClick={() => select.mutate(evaluation)}>{select.isPending && <Spinner />}Vào lượt thi</Button>
      {runs.isError && <p role="alert" className="image-error">{runs.error.message}</p>}
    </> : <form className="stack" onSubmit={e => { e.preventDefault(); setAccess(true); login.mutate({ server, username: mode === 'password' ? username : '', password: mode === 'password' ? password : '', session_id: mode === 'session' ? token : '' }) }}>
      <label className="field">Server URL<input type="url" required placeholder="https://dres.example.org" value={server} onChange={e => setServer(e.target.value)} autoComplete="url" /></label>
      <label className="field">Cách đăng nhập<select value={mode} onChange={e => { setMode(e.target.value); setPassword(''); setToken('') }}><option value="password">Username + password</option><option value="session">Session ID</option></select></label>
      {mode === 'password' ? <><label className="field">Username<input required value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" /></label><label className="field">Password<input type="password" required value={password} onChange={e => setPassword(e.target.value)} autoComplete="current-password" /></label></> : <label className="field">Session ID<input type="password" required value={token} onChange={e => setToken(e.target.value)} autoComplete="off" /></label>}
      <Button className="primary" type="submit" disabled={login.isPending}>{login.isPending ? <Spinner /> : <LogIn size={16} />}Đăng nhập DRES</Button>
    </form>}
    {[login.error, select.error, logout.error].filter(Boolean).map((error, i) => <p key={i} role="alert" className="image-error">{error?.message}</p>)}
    <Button onClick={() => { setPractice(true); setAccess(false); sessionStorage.setItem('image-practice', 'true') }}>Luyện tập</Button>
  </section></main>
}

export function DRESSubmit({ imageID, session }: { imageID: string; session: DRESSession }) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState(imageID)
  const [collection, setCollection] = useState('')
  const [task, setTask] = useState('')
  const submit = useMutation({ mutationFn: imageApi.dresSubmit })
  return <Dialog.Root open={open} onOpenChange={setOpen}>
    <Button disabled={!session.connected || !session.evaluation_id} onClick={() => { setName(imageID); submit.reset(); setOpen(true) }} title={session.connected ? 'Gửi ảnh đến DRES' : 'Đăng nhập DRES để gửi bài'}><Send size={14} />Gửi DRES</Button>
    <Dialog.Portal><Dialog.Overlay className="dialog-overlay" /><Dialog.Content className="dialog-content dres-submit stack">
      <Dialog.Title>Gửi ảnh đến DRES</Dialog.Title><Dialog.Description>{session.server} · {session.username} · Evaluation {session.evaluation_id}</Dialog.Description>
      <img src={`/api/images/media/${encodeURIComponent(imageID)}?thumbnail=true`} alt={imageID} />
      <label className="field">Media item<input value={name} onChange={e => setName(e.target.value)} /></label>
      <label className="field">Collection<input value={collection} onChange={e => setCollection(e.target.value)} /></label>
      <label className="field">Task ID<input placeholder="Task hiện tại" value={task} onChange={e => setTask(e.target.value)} /></label>
      <details><summary>Xem payload</summary><pre>{JSON.stringify({ answerSets: [{ ...(task.trim() ? { taskId: task.trim() } : {}), answers: [{ mediaItemName: name.trim(), ...(collection.trim() ? { mediaItemCollectionName: collection.trim() } : {}) }] }] }, null, 2)}</pre></details>
      {submit.isError && <p role="alert" className="image-error">{submit.error.message}</p>}
      {submit.data && <p role="status">HTTP {submit.data.http_status} · {submit.data.submission ?? (submit.data.http_status === 202 ? 'Chờ chấm' : 'Đã nhận')} · {submit.data.description}</p>}
      <div className="row"><Button className="primary" disabled={!name.trim() || submit.isPending || submit.isSuccess} onClick={() => submit.mutate({ image_id: imageID, media_item_name: name.trim(), collection, task_id: task })}>{submit.isPending ? <Spinner /> : <Send size={14} />}Gửi ảnh này</Button><Dialog.Close asChild><Button>Đóng</Button></Dialog.Close></div>
    </Dialog.Content></Dialog.Portal>
  </Dialog.Root>
}
