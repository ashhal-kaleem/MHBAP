/**
 * SessionPanel — session lifecycle control + history list.
 *
 * Features:
 *  - Start / End session
 *  - Per-session: delete, rename context, CSV export
 *  - Inline stats card for completed sessions
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { clsx } from 'clsx'
import { Download, Edit2, History, Play, Square, Trash2, X } from 'lucide-react'
import {
  createSession,
  deleteSession,
  endSession,
  exportSessionCsv,
  listUserSessions,
  updateSessionContext,
  startRunner,
  stopRunner,
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
  } catch {
    return iso
  }
}

const STATUS_STYLE: Record<Session['status'], string> = {
  active:    'bg-green-500/15 text-green-400',
  completed: 'bg-gray-500/15 text-gray-400',
  error:     'bg-red-500/15 text-red-400',
}

export function SessionPanel({ user, activeSession, onSelectSession }: SessionPanelProps) {
  const [sessions, setSessions] = useState<Session[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Context editing state
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  // Stats for active/selected completed session
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

  // Focus the rename input when it opens
  useEffect(() => {
    if (editingId) editInputRef.current?.focus()
  }, [editingId])

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
      try {
        await stopRunner(activeSession.id)
      } catch (runnerErr) {
        console.error('Failed to stop runner:', runnerErr)
      }
      const ended = await endSession(activeSession.id)
      setSessions((prev) => prev.map((s) => (s.id === ended.id ? ended : s)))
      onSelectSession(ended)
      refetchStats()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to end session')
    } finally { setBusy(false) }
  }

  const handleDelete = async (s: Session) => {
    if (!window.confirm(`Delete session from ${formatDateTime(s.started_at)}? This also removes all predictions.`)) return
    setBusy(true); setError(null)
    try {
      await deleteSession(s.id)
      setSessions((prev) => prev.filter((x) => x.id !== s.id))
      if (activeSession?.id === s.id) onSelectSession(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed')
    } finally { setBusy(false) }
  }

  const startEditing = (s: Session) => {
    setEditingId(s.id)
    setEditValue(s.context)
  }

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
    try {
      await exportSessionCsv(sessionId)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
    } finally { setBusy(false) }
  }

  const isLive = activeSession?.status === 'active'

  return (
    <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm p-6">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-gray-900">Session</p>
        <div className="flex items-center gap-2">
          {!isLive ? (
            <button
              onClick={handleStart}
              disabled={!user || busy}
              className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Play className="h-3.5 w-3.5" /> Start Analysis
            </button>
          ) : (
            <>
              <span className="inline-flex items-center gap-1 rounded-full bg-green-500/10 px-2 py-0.5 text-[10px] font-medium text-green-400">
                🎥 Analysis active
              </span>
              <button
                onClick={handleEnd}
                disabled={busy}
                className="inline-flex items-center gap-1.5 rounded-lg bg-red-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Square className="h-3.5 w-3.5" /> Stop Analysis
              </button>
            </>
          )}

        </div>
      </div>

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}

      {/* History list */}
      <div className="mt-4">
        <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-gray-500">
          <History className="h-3.5 w-3.5" /> History
        </p>
        {!user && <p className="text-xs text-gray-500">Loading user…</p>}
        {user && sessions.length === 0 && (
          <div className="text-sm text-gray-500 italic p-6 text-center border border-dashed border-gray-200 rounded-xl bg-gray-50/50 mt-2">No sessions yet — start one above.</div>
        )}
        <ul className="max-h-52 space-y-1 overflow-y-auto pr-1">
          {sessions.map((s) => (
            <li key={s.id}>
              {editingId === s.id ? (
                /* Inline rename editor */
                <div className="flex items-center gap-1 rounded-lg bg-white border border-gray-200 px-2 py-1 shadow-sm">
                  <input
                    ref={editInputRef}
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') commitContext(s.id)
                      if (e.key === 'Escape') cancelEditing()
                    }}
                    className="min-w-0 flex-1 bg-transparent text-xs text-gray-900 outline-none placeholder:text-gray-400"
                    placeholder="Context label…"
                    maxLength={120}
                  />
                  <button onClick={() => commitContext(s.id)} className="text-green-500 hover:text-green-600 text-xs px-1">✓</button>
                  <button onClick={cancelEditing} className="text-gray-400 hover:text-gray-600"><X className="h-3 w-3" /></button>
                </div>
              ) : (
                /* Normal row */
                <div
                  className={clsx(
                    'group flex w-full items-center gap-2 rounded-lg px-2 py-1.5 transition',
                    activeSession?.id === s.id ? 'bg-blue-50/80 border border-blue-100 shadow-sm' : 'hover:bg-gray-50/80 border border-transparent',
                  )}
                >
                  <button
                    onClick={() => onSelectSession(s)}
                    className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  >
                    <span className={clsx('text-xs', activeSession?.id === s.id ? 'text-blue-900 font-medium' : 'text-gray-500')}>
                      <span className="font-mono">{formatDateTime(s.started_at)}</span>
                      {s.context && s.context !== 'unspecified' && s.context !== 'live' && (
                        <span className="ml-1.5 text-gray-400">· {s.context}</span>
                      )}
                    </span>
                    <span className={clsx('ml-auto shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium', STATUS_STYLE[s.status])}>
                      {s.status === 'active' ? '🎥 active' : s.status}
                    </span>
                  </button>

                  {/* Action buttons — visible on hover or when row is active */}
                  <div className={clsx('flex shrink-0 items-center gap-0.5', activeSession?.id === s.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')}>
                    <button
                      title="Rename context"
                      onClick={() => startEditing(s)}
                      className="rounded p-1 text-gray-400 hover:bg-white hover:text-gray-900 hover:shadow-sm border border-transparent hover:border-gray-200"
                    >
                      <Edit2 className="h-3 w-3" />
                    </button>
                    {s.status === 'completed' && (
                      <button
                        title="Export CSV"
                        onClick={() => handleExport(s.id)}
                        disabled={busy}
                        className="rounded p-1 text-gray-400 hover:bg-white hover:text-gray-900 hover:shadow-sm border border-transparent hover:border-gray-200 disabled:opacity-40"
                      >
                        <Download className="h-3 w-3" />
                      </button>
                    )}
                    <button
                      title="Delete session"
                      onClick={() => handleDelete(s)}
                      disabled={busy}
                      className="rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-600 hover:border-red-100 border border-transparent disabled:opacity-40"
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
      {stats && <SessionStatsCard stats={stats} loading={statsLoading} />}
      {completedId && !stats && statsLoading && (
        <p className="mt-3 text-xs text-gray-500">Loading stats…</p>
      )}
    </div>
  )
}
