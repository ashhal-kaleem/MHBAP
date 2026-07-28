import { useCallback, useEffect, useRef, useState } from 'react'
import type { ConnectionStatus, WsMessage } from '@/types'

const WS_URL = import.meta.env.VITE_WS_URL ?? `ws://${window.location.hostname}:8000/api/v1/stream/ws`
const RECONNECT_DELAY_MS = 3000
const MAX_HISTORY = 120   // ~2 min at 1 msg/s

export function useStream(sessionId: string | null) {
  const [status, setStatus] = useState<ConnectionStatus>('closed')
  const [latest, setLatest] = useState<WsMessage | null>(null)
  const [history, setHistory] = useState<WsMessage[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    if (!sessionId) return
    const url = `${WS_URL}/${sessionId}`
    setStatus('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => setStatus('open')

    ws.onmessage = (evt) => {
      try {
        const msg: WsMessage = JSON.parse(evt.data as string)
        setLatest(msg)
        setHistory((prev) => [...prev.slice(-(MAX_HISTORY - 1)), msg])
      } catch {
        // ignore malformed frames
      }
    }

    ws.onerror = () => setStatus('error')

    ws.onclose = () => {
      setStatus('closed')
      // auto-reconnect
      timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS)
    }
  }, [sessionId])

  useEffect(() => {
    connect()
    return () => {
      timerRef.current && clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const disconnect = useCallback(() => {
    timerRef.current && clearTimeout(timerRef.current)
    wsRef.current?.close()
  }, [])

  return { status, latest, history, disconnect }
}
