import { useEffect, useState } from 'react'
import { listSessionPredictions } from '@/services/api'
import type { WsMessage } from '@/types'

/**
 * Fetches all predictions for a session over REST and shapes them as
 * WsMessage[] so the Dashboard can render historical (non-live) sessions
 * through the same rendering path as the live WebSocket stream.
 */
export function useHistoricalPredictions(sessionId: string | null) {
  const [history, setHistory] = useState<WsMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!sessionId) {
      setHistory([])
      return
    }
    let cancelled = false

    async function load() {
      setLoading(true)
      setError(null)
      try {
        const predictions = await listSessionPredictions(sessionId!)
        if (cancelled) return
        setHistory(predictions.map((payload) => ({ type: 'prediction' as const, payload })))
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load history')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [sessionId])

  return { history, loading, error }
}
