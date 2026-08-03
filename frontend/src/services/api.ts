import type { Prediction, Session, SessionStats, User, UserAnalytics, XAISummary } from '@/types'

const BASE = '/api/v1'

// ── Token store ──────────────────────────────────────────────────────────────
// Written by useCurrentUser after login/register; read here for every request.
const TOKEN_KEY = 'mhbap_token'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken()
  const authHeader: Record<string, string> = token
    ? { Authorization: `Bearer ${token}` }
    : {}
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...authHeader,
      ...(init?.headers as Record<string, string> | undefined),
    },
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json() as Promise<T>
}

// Users
export const createUser = (username: string, email: string, role = 'participant') =>
  json<User>('/users/', {
    method: 'POST',
    body: JSON.stringify({ username, email, role }),
  })

export const getUser = (userId: string) => json<User>(`/users/${userId}`)

// Sessions
export const createSession = (userId: string, context = 'unspecified') =>
  json<Session>('/sessions/', {
    method: 'POST',
    body: JSON.stringify({ user_id: userId, context }),
  })

export const getSession = (sessionId: string) => json<Session>(`/sessions/${sessionId}`)

export const listUserSessions = (userId: string, limit = 50) =>
  json<Session[]>(`/sessions/user/${userId}?limit=${limit}`)

export const endSession = (sessionId: string) =>
  json<Session>(`/sessions/${sessionId}/end`, { method: 'POST' })

export const deleteSession = (sessionId: string) =>
  fetch(`${BASE}/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: getToken() ? { Authorization: `Bearer ${getToken()!}` } : {},
  }).then((res) => {
    if (!res.ok && res.status !== 204) throw new Error(`API ${res.status}`)
  })

export const updateSessionContext = (sessionId: string, context: string) =>
  json<Session>(`/sessions/${sessionId}/context`, {
    method: 'PATCH',
    body: JSON.stringify({ context }),
  })

export const getSessionStats = (sessionId: string) =>
  json<SessionStats>(`/sessions/${sessionId}/stats`)

/** Triggers CSV download via hidden anchor element. */
export async function exportSessionCsv(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/sessions/${sessionId}/export/csv`, {
    headers: getToken() ? { Authorization: `Bearer ${getToken()!}` } : {},
  })
  if (!res.ok) throw new Error(`Export failed: ${res.status}`)
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `mhbap_session_${sessionId}.csv`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

// Predictions
export const listSessionPredictions = (sessionId: string, limit = 1000) =>
  json<Prediction[]>(`/predictions/session/${sessionId}?limit=${limit}`)

export const latestSessionPrediction = (sessionId: string) =>
  json<Prediction>(`/predictions/session/${sessionId}/latest`)

export const getXAISummary = (sessionId: string) =>
  json<XAISummary>(`/predictions/session/${sessionId}/xai`)

// Analytics (Phase 10)
export const getUserAnalytics = (userId: string) =>
  json<UserAnalytics>(`/analytics/user/${userId}`)

export async function exportSessionJson(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/sessions/${sessionId}/export/json`, {
    headers: getToken() ? { Authorization: `Bearer ${getToken()!}` } : {},
  })
  if (!res.ok) throw new Error(`Export failed: ${res.status}`)
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `mhbap_session_${sessionId}.json`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export async function exportUserCsv(userId: string): Promise<void> {
  const res = await fetch(`${BASE}/analytics/user/${userId}/export/csv`, {
    headers: getToken() ? { Authorization: `Bearer ${getToken()!}` } : {},
  })
  if (!res.ok) throw new Error(`Export failed: ${res.status}`)
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `mhbap_user_${userId}_all_sessions.csv`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

// Health
export const getHealth = () =>
  json<{ status: string; db: string; redis: string }>('/health/ready')

// Runner Control
export const startRunner = (sessionId: string) =>
  json<{ session_id: string; running: boolean }>(`/runner/session/${sessionId}/start`, { method: 'POST' })

export const stopRunner = (sessionId: string) =>
  json<{ session_id: string; running: boolean }>(`/runner/session/${sessionId}/stop`, { method: 'POST' })
