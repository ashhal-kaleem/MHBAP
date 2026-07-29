/**
 * useUserAnalytics — fetches cross-session analytics for the current user.
 * Polls every 60 s when a user is active; stops if analytics is null.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { getUserAnalytics } from '@/services/api'
import type { UserAnalytics } from '@/types'

const POLL_MS = 60_000

export function useUserAnalytics(userId: string | null) {
  const [analytics, setAnalytics] = useState<UserAnalytics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetch_ = useCallback(async () => {
    if (!userId) return
    try {
      const data = await getUserAnalytics(userId)
      setAnalytics(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analytics fetch failed')
    } finally {
      setLoading(false)
    }
  }, [userId])

  useEffect(() => {
    if (!userId) {
      setAnalytics(null)
      return
    }
    setLoading(true)
    void fetch_()
    timerRef.current = setInterval(fetch_, POLL_MS)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [userId, fetch_])

  return { analytics, loading, error, refresh: fetch_ }
}
