/**
 * SessionPanel — session lifecycle control + history list.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { clsx } from 'clsx'
import { Download, Edit2, History, Play, Square, Trash2, X, Clock, AlertCircle } from 'lucide-react'
import {
  createSession, deleteSession, endSession, exportSessionCsv,
  listUserSessions, updateSessionContext, startRunner, stopRunner,
} from '@/services/api'
import { useSessionStats } from '@/hooks/useSessionStats'
import { SessionStatsCard } from './SessionStatsCard'
import type { Session, User } from '@/types'

interface SessionPanelProps {
  user: User | null
  activeSession: Session | null
  onSelectSession: (session: Session | null) => void
}

function formatDateTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString('en-US', {
      month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
    })
  } catch { return iso }
}

function formatDuration(start: string, end: string | null): string {
  if (!end) return ''
  const secs = Math.round((new Date(end).getTime() - new Date(start).getTime()) / 1000)
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

const STATUS_PILL: Record<Session['status'], string> = {
  active:    'bg-green-100 text-green-700 border-green-200',
  completed: 'bg-gray-100 text-gray-500 border-gray-200',
  error:     'bg-red-100 text-red-600 border-red-200',
}

export function SessionPanel({ user, activeSession, onSelectSession }: SessionPanelProps) {
  const [sessions, setSessions] = useState<Session[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  const completedId = activeSession?.status === 'completed' ? activeSession.id : null
  const { stats, loading: statsLoading, refetch: refetchStats } = useSessionStats(completedId)

  const refresh = useCallback(async () => {
    if (!user) return
    try {
      const list = await listUserSessions(user.id)
      setSessions(list)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions')
    }
  }, [user])

  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { if (editingId) editInputRef.current?.focus() }, [editingId])

  const handleStart = async () => {
    if (!user || busy) return
    setBusy(true); setError(null)
    try {
      const session = await createSession(user.id, 'live')
      setSessions((prev) => [session, ...prev])
      onSelectSession(session)
      await startRunner(session.id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start session')
    } finally { setBusy(false) }
  }

  const handleEnd = async () => {
    if (!activeSession || busy) return
    setBusy(true); setError(null)
    try {
      try { await stopRunner(activeSession.id) } catch (e) { console.error('Runner stop:', e) }
      const ended = await endSession(activeSession.id)
      setSessions((prev) => prev.map((s) => (s.id === ended.id ? ended : s)))
      onSelectSession(ended)
      refetchStats()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to end session')
    } finally { setBusy(false) }
  }

  const handleDelete = async (s: Session) => {
    if (!window.confirm(`Delete session from ${formatDateTime(s.started_at)}? This removes all predictions.`)) return
    setBusy(true); setError(null)
    try {
      await deleteSession(s.id)
      setSessions((prev) => prev.filter((x) => x.id !== s.id))
      if (activeSession?.id === s.id) onSelectSession(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed')
    } finally { setBusy(false) }
  }

  const startEditing = (s: Session) => { setEditingId(s.id); setEditValue(s.context) }
  const cancelEditing = () => setEditingId(null)

  const commitContext = async (sessionId: string) => {
    const trimmed = editValue.trim()
    if (!trimmed) { cancelEditing(); return }
    setBusy(true); setError(null)
    try {
      const updated = await updateSessionContext(sessionId, trimmed)
      setSessions((prev) => prev.map((s) => (s.id === updated.id ? updated : s)))
      if (activeSession?.id === updated.id) onSelectSession(updated)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rename failed')
    } finally { setBusy(false); setEditingId(null) }
  }

  const handleExport = async (sessionId: string) => {
    setBusy(true); setError(null)
    try { await exportSessionCsv(sessionId) }
    catch (err) { setError(err instanceof Error ? err.message : 'Export failed') }
    finally { setBusy(false) }
  }

  const isLive = activeSession?.status === 'active'

  return (
    <div className="rounded-2xl bg-white border border-gray-100 shadow-sm p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm font-semibold text-gray-900">Session Control</p>
        <div className="flex items-center gap-2">
          {isLive && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-green-50 border border-green-200 px-2.5 py-1 text-xs font-semibold text-green-700">
              <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
              Analysis active
            </span>
          )}
          {!isLive ? (
            <button
              onClick={handleStart}
              disabled={!user || busy}
              className="inline-flex items-center gap-1.5 rounded-lg bg-plum px-4 py-1.5 text-xs font-semibold text-white transition hover:bg-plum-dark disabled:cursor-not-allowed disabled:opacity-40 shadow-sm"
            >
              {busy ? (
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              Start Analysis
            </button>
          ) : (
            <button
              onClick={handleEnd}
              disabled={busy}
              className="inline-flex items-center gap-1.5 rounded-lg bg-red-600 px-4 py-1.5 text-xs font-semibold text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-40 shadow-sm"
            >
              {busy ? (
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Square className="h-3.5 w-3.5" />
              )}
              Stop Analysis
            </button>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-3 flex items-center gap-2 rounded-lg bg-red-50 border border-red-100 px-3 py-2 text-xs text-red-600">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          {error}
        </div>
      )}

      {/* History */}
      <div>
        <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          <History className="h-3 w-3" /> History
        </p>

        {!user && (
          <p className="text-xs text-gray-400 py-2">Authenticating…</p>
        )}

        {user && sessions.length === 0 && (
          <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-gray-200 bg-gray-50/50 py-8 text-center">
            <Clock className="h-6 w-6 text-gray-300" />
            <p className="text-xs text-gray-400">No sessions yet — start one above.</p>
          </div>
        )}

        <ul className="max-h-64 space-y-0.5 overflow-y-auto pr-0.5 mt-1">
          {sessions.map((s) => (
            <li key={s.id}>
              {editingId === s.id ? (
                <div className="flex items-center gap-1.5 rounded-lg bg-white border border-plum/30 px-3 py-2 shadow-sm">
                  <input
                    ref={editInputRef}
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') commitContext(s.id)
                      if (e.key === 'Escape') cancelEditing()
                    }}
                    className="min-w-0 flex-1 bg-transparent text-xs text-gray-900 outline-none placeholder:text-gray-300"
                    placeholder="Context label…"
                    maxLength={120}
                  />
                  <button onClick={() => commitContext(s.id)} className="text-green-500 hover:text-green-600 text-xs font-bold px-1">✓</button>
                  <button onClick={cancelEditing} className="text-gray-300 hover:text-gray-500"><X className="h-3.5 w-3.5" /></button>
                </div>
              ) : (
                <div
                  className={clsx(
                    'group flex w-full items-center gap-2 rounded-lg px-2.5 py-2 transition-all cursor-pointer',
                    activeSession?.id === s.id
                      ? 'bg-plum/5 border border-plum/20 shadow-sm'
                      : 'border border-transparent hover:bg-gray-50',
                  )}
                  onClick={() => onSelectSession(s)}
                >
                  <div className="flex min-w-0 flex-1 items-center gap-2">
                    <div className="min-w-0">
                      <p className={clsx(
                        'text-xs truncate',
                        activeSession?.id === s.id ? 'font-semibold text-plum' : 'text-gray-700',
                      )}>
                        {formatDateTime(s.started_at)}
                      </p>
                      {s.context && s.context !== 'unspecified' && s.context !== 'live' && (
                        <p className="text-[11px] text-gray-400 truncate">{s.context}</p>
                      )}
                      {s.ended_at && (
                        <p className="text-[11px] text-gray-300">{formatDuration(s.started_at, s.ended_at)}</p>
                      )}
                    </div>
                    <span className={clsx('ml-auto shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold capitalize', STATUS_PILL[s.status])}>
                      {s.status}
                    </span>
                  </div>

                  {/* Actions */}
                  <div
                    className={clsx(
                      'flex shrink-0 items-center gap-0.5 transition-opacity',
                      activeSession?.id === s.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100',
                    )}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      title="Rename context"
                      onClick={() => startEditing(s)}
                      className="rounded p-1.5 text-gray-300 hover:bg-white hover:text-gray-700 hover:shadow-sm border border-transparent hover:border-gray-200 transition-all"
                    >
                      <Edit2 className="h-3 w-3" />
                    </button>
                    {s.status === 'completed' && (
                      <button
                        title="Export CSV"
                        onClick={() => handleExport(s.id)}
                        disabled={busy}
                        className="rounded p-1.5 text-gray-300 hover:bg-white hover:text-gray-700 hover:shadow-sm border border-transparent hover:border-gray-200 transition-all disabled:opacity-30"
                      >
                        <Download className="h-3 w-3" />
                      </button>
                    )}
                    <button
                      title="Delete session"
                      onClick={() => handleDelete(s)}
                      disabled={busy}
                      className="rounded p-1.5 text-gray-300 hover:bg-red-50 hover:text-red-500 hover:border-red-100 border border-transparent transition-all disabled:opacity-30"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>

      {/* Stats card for completed sessions */}
      {completedId && statsLoading && !stats && (
        <div className="mt-4 flex items-center gap-2 text-xs text-gray-400">
          <div className="w-3.5 h-3.5 border-2 border-plum/30 border-t-plum rounded-full animate-spin" />
          Loading stats…
        </div>
      )}
      {stats && <SessionStatsCard stats={stats} loading={statsLoading} />}
    </div>
  )
}
