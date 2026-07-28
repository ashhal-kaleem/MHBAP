import { useCallback, useEffect, useRef, useState } from 'react'
import type { ConnectionStatus, WsMessage } from '@/types'

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
  const wsRef = useRef<WebSocket | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    if (!sessionId) return

    const path =
      sessionId === 'demo'
        ? '/api/v1/stream/demo'
        : `/api/v1/stream/session/${sessionId}`
    const url = `${WS_BASE}${path}`

    setStatus('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => setStatus('open')

    ws.onmessage = (evt: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(evt.data) as WsMessage
        setLatest(msg)
        setHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), msg])
      } catch {
        // ignore malformed frames
      }
    }

    ws.onerror = () => setStatus('error')

    ws.onclose = () => {
      setStatus('closed')
      timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS)
    }
  }, [sessionId])

  useEffect(() => {
    connect()
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const disconnect = useCallback(() => {
    if (timerRef.current !== null) clearTimeout(timerRef.current)
    wsRef.current?.close()
  }, [])

  return { status, latest, history, disconnect }
}
