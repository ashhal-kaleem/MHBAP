import { useEffect, useState } from 'react'
import { createUser, getUser } from '@/services/api'
import type { User } from '@/types'

const STORAGE_KEY = 'mhbap_user_id'

function randomSuffix(): string {
  return Math.random().toString(36).slice(2, 8)
}

/**
 * Ensures a User record exists for this browser and returns it.
 * No auth in Phase 2/7 scope — this provisions a lightweight
 * "participant" identity the first time the app loads, then reuses
 * the same user id from localStorage on every subsequent visit.
 */
export function useCurrentUser() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function bootstrap() {
      setLoading(true)
      setError(null)
      try {
        const existingId = localStorage.getItem(STORAGE_KEY)
        if (existingId) {
          try {
            const found = await getUser(existingId)
            if (!cancelled) setUser(found)
            return
          } catch {
            // stale id (e.g. dev DB was reset) — fall through and re-provision
            localStorage.removeItem(STORAGE_KEY)
          }
        }

        const suffix = randomSuffix()
        const created = await createUser(`guest-${suffix}`, `guest-${suffix}@mhbap.local`)
        localStorage.setItem(STORAGE_KEY, created.id)
        if (!cancelled) setUser(created)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load user')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    bootstrap()
    return () => {
      cancelled = true
    }
  }, [])

  return { user, loading, error }
}
