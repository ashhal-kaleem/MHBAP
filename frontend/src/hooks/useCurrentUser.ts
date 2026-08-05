import { useEffect, useState } from 'react'
import type { User } from '@/types'

const USER_ID_KEY = 'mhbap_user_id'
const TOKEN_KEY = 'mhbap_token'
const EMAIL_KEY = 'mhbap_email'
const PASSWORD_KEY = 'mhbap_password'

const BASE = '/api/v1'

function randomSuffix(): string {
  return Math.random().toString(36).slice(2, 8)
}

/** Generate a password that satisfies the backend strength validator:
 *  ≥ 8 chars, at least one digit or special char. */
function generatePassword(): string {
  return `Mhbap-${Math.random().toString(36).slice(2, 10)}!1`
}

/**
 * Ensures an authenticated User record exists for this browser session.
 *
 * Flow:
 *  1. If a token is already stored, call GET /auth/me to verify it and
 *     return the profile. If the token is expired/invalid, fall through.
 *  2. If stored credentials (email + password) exist, call POST /auth/login
 *     to obtain a fresh token and return the profile.
 *  3. Otherwise register a brand-new guest account via POST /auth/register,
 *     persist the credentials and token, then return the profile.
 *
 * The token is stored under 'mhbap_token' in localStorage.
 * api.ts reads that key and attaches it as a Bearer header to every
 * authenticated request.
 */
export function useCurrentUser() {
  const [user, setUser] = useState<CurrentUser | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function bootstrap() {
      setLoading(true)
      setError(null)

      try {
        // ── Step 1: try the cached token ──────────────────────────────────
        const cachedToken = localStorage.getItem(TOKEN_KEY)
        if (cachedToken) {
          try {
            const res = await fetch(`${BASE}/auth/me`, {
              headers: { Authorization: `Bearer ${cachedToken}` },
            })
            if (res.ok) {
              const me = await res.json()
              // /auth/me returns { user_id, email, display_name, role }
              // We need the full User shape — fetch it.
              const userId = localStorage.getItem(USER_ID_KEY) ?? me.user_id
              const userRes = await fetch(`${BASE}/users/${userId}`, {
                headers: { Authorization: `Bearer ${cachedToken}` },
              })
              if (userRes.ok) {
                if (!cancelled) setUser(await userRes.json())
                return
              }
            }
            // Token invalid / expired — fall through to login/register
          } catch {
            // network blip — fall through
          }
          localStorage.removeItem(TOKEN_KEY)
        }

        // ── Step 2: re-login with stored credentials ──────────────────────
        const storedEmail = localStorage.getItem(EMAIL_KEY)
        const storedPassword = localStorage.getItem(PASSWORD_KEY)
        if (storedEmail && storedPassword) {
          try {
            const loginRes = await fetch(`${BASE}/auth/login`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ email: storedEmail, password: storedPassword }),
            })
            if (loginRes.ok) {
              const { access_token, user_id } = await loginRes.json()
              localStorage.setItem(TOKEN_KEY, access_token)
              localStorage.setItem(USER_ID_KEY, user_id)
              const userRes = await fetch(`${BASE}/users/${user_id}`, {
                headers: { Authorization: `Bearer ${access_token}` },
              })
              if (userRes.ok && !cancelled) {
                setUser(await userRes.json())
                return
              }
            }
          } catch {
            // fall through to fresh register
          }
          // Credentials no longer valid (DB was reset etc.) — clear and re-register
          localStorage.removeItem(EMAIL_KEY)
          localStorage.removeItem(PASSWORD_KEY)
          localStorage.removeItem(USER_ID_KEY)
        }

        // ── Step 3: register a fresh guest account ────────────────────────
        const suffix = randomSuffix()
        const email = `guest-${suffix}@mhbap.example`
        const password = generatePassword()

        const regRes = await fetch(`${BASE}/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password, display_name: `Guest ${suffix}` }),
        })
        if (!regRes.ok) {
          const body = await regRes.text()
          throw new Error(`Registration failed (${regRes.status}): ${body}`)
        }
        const { access_token, user_id } = await regRes.json()

        localStorage.setItem(TOKEN_KEY, access_token)
        localStorage.setItem(USER_ID_KEY, user_id)
        localStorage.setItem(EMAIL_KEY, email)
        localStorage.setItem(PASSWORD_KEY, password)

        const userRes = await fetch(`${BASE}/users/${user_id}`, {
          headers: { Authorization: `Bearer ${access_token}` },
        })
        if (userRes.ok && !cancelled) setUser(await userRes.json())
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load user')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    bootstrap()
    return () => { cancelled = true }
  }, [])

  return { user, loading, error }
}
