import type { Prediction, Session, SessionStats, User } from '@/types'

const BASE = '/api/v1'

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
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
  fetch(`${BASE}/sessions/${sessionId}`, { method: 'DELETE' }).then((res) => {
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
  const res = await fetch(`${BASE}/sessions/${sessionId}/export/csv`)
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

// Health
export const getHealth = () =>
  json<{ status: string; db: string; redis: string }>('/health/ready')
