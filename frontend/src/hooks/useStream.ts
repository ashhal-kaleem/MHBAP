import { useCallback, useEffect, useRef, useState } from 'react'
import type { ConnectionStatus, WsMessage } from '@/types'
import { getToken } from '@/services/api'

/**
 * WS base URL.
 * - VITE_WS_URL env var overrides (e.g. for production).
 * - Default uses the same host/port as the page, hitting the Vite proxy
 *   which forwards /api/... WebSocket upgrades to localhost:8000.
 */
const WS_BASE =
  (import.meta.env.VITE_WS_URL as string | undefined) ??
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`

const RECONNECT_DELAY_MS = 3000
const MAX_HISTORY = 120   // ~2 min at 1 msg/s

/**
 * sessionId:
 *   "demo"     → connects to /api/v1/stream/demo
 *   <uuid str> → connects to /api/v1/stream/session/<uuid>
 *   null       → no connection
 */
export function useStream(sessionId: string | null) {
  const [status, setStatus] = useState<ConnectionStatus>('closed')
  const [latest, setLatest] = useState<WsMessage | null>(null)
  const [history, setHistory] = useState<WsMessage[]>([])

  const wsRef        = useRef<WebSocket | null>(null)
  const timerRef     = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Stores the deactivator for the currently live socket so that BOTH
  // cleanup paths (unmount and sessionId change) can disable stale callbacks.
  const deactivateRef = useRef<(() => void) | null>(null)

  // Keep sessionId in a ref so `connect` never changes identity.
  const sessionIdRef = useRef(sessionId)
  sessionIdRef.current = sessionId

  /**
   * Tear down whatever is currently open — socket + timer + active flag.
   * Safe to call multiple times (idempotent).
   */
  const teardown = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    // Deactivate the live socket's callbacks BEFORE closing so that the
    // onclose handler cannot schedule a new reconnect timer.
    if (deactivateRef.current) {
      deactivateRef.current()
      deactivateRef.current = null
    }
    if (wsRef.current) {
      const old = wsRef.current
      old.onopen = null
      old.onmessage = null
      old.onerror = null
      old.onclose = null
      old.close()
      wsRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    const sid = sessionIdRef.current
    if (!sid) return

    // Always tear down first so we never have two live sockets.
    teardown()

    const path =
      sid === 'demo'
        ? '/api/v1/stream/demo'
        : `/api/v1/stream/session/${sid}`
    let url = `${WS_BASE}${path}`

    // Browser WebSockets cannot send Authorization headers.
    // Pass the token via query param as expected by get_ws_current_user.
    const token = getToken()
    if (token) url += `?access_token=${token}`

    setStatus('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    // Per-socket active guard: prevents callbacks from a replaced socket
    // from mutating shared state after teardown() has run.
    let active = true
    deactivateRef.current = () => { active = false }

    ws.onopen = () => {
      if (!active) return
      setStatus('open')
    }

    ws.onmessage = (evt: MessageEvent<string>) => {
      if (!active) return
      try {
        const msg = JSON.parse(evt.data) as WsMessage
        // Skip ping frames entirely — they carry no prediction data and
        // would trigger unnecessary Dashboard re-renders every 25 seconds.
        if (msg.type === 'ping') return
        setLatest(msg)
        setHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), msg])
      } catch {
        // ignore malformed frames
      }
    }

    ws.onerror = () => {
      if (!active) return
      setStatus('error')
    }

    ws.onclose = () => {
      if (!active) return
      setStatus('closed')
      timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS)
    }
  }, [teardown])  // teardown is stable (useCallback with [])


  // ── Mount / unmount ───────────────────────────────────────────────────────
  useEffect(() => {
    connect()
    return () => { teardown() }
  }, [connect, teardown])

  // ── sessionId change (demo ↔ live ↔ null) ────────────────────────────────
  // Skip the initial render because the effect above already called connect().
  const prevSessionIdRef = useRef(sessionId)
  useEffect(() => {
    if (prevSessionIdRef.current === sessionId) return
    prevSessionIdRef.current = sessionId

    // teardown() deactivates the old socket so its onclose won't fire.
    teardown()
    setHistory([])
    setLatest(null)

    if (sessionId) connect()
    else setStatus('closed')
  }, [sessionId, connect, teardown])

  const disconnect = useCallback(() => { teardown() }, [teardown])

  return { status, latest, history, disconnect }
}
