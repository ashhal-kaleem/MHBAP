/**
 * useSessionStats — fetches aggregated stats for a completed session.
 * Returns null while loading or when sessionId is null (demo mode).
 */
import { useEffect, useState } from 'react'
import { getSessionStats } from '@/services/api'
import type { SessionStats } from '@/types'

interface UseSessionStatsResult {
  stats: SessionStats | null
  loading: boolean
  error: string | null
  refetch: () => void
}

export function useSessionStats(sessionId: string | null): UseSessionStatsResult {
  const [stats, setStats] = useState<SessionStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    if (!sessionId) {
      setStats(null)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    getSessionStats(sessionId)
      .then((s) => { if (!cancelled) setStats(s) })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [sessionId, tick])

  return { stats, loading, error, refetch: () => setTick((t) => t + 1) }
}
