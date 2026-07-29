/**
 * useXAISummary — fetches session-level XAI aggregation (avg SHAP weights + trends).
 * Auto-fetches when sessionId changes. Re-fetches on refetch().
 */
import { useEffect, useState } from 'react'
import { getXAISummary } from '@/services/api'
import type { XAISummary } from '@/types'

interface UseXAISummaryResult {
  summary: XAISummary | null
  loading: boolean
  error: string | null
  refetch: () => void
}

export function useXAISummary(sessionId: string | null): UseXAISummaryResult {
  const [summary, setSummary] = useState<XAISummary | null>(null)
  const [loading, setLoading]  = useState(false)
  const [error,   setError]    = useState<string | null>(null)
  const [tick,    setTick]     = useState(0)

  useEffect(() => {
    if (!sessionId) { setSummary(null); return }
    let cancelled = false
    setLoading(true)
    setError(null)
    getXAISummary(sessionId)
      .then((s) => { if (!cancelled) setSummary(s) })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [sessionId, tick])

  return { summary, loading, error, refetch: () => setTick((t) => t + 1) }
}
