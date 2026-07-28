import type { Prediction, Session } from '@/types'

const BASE = '/api/v1'

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json() as Promise<T>
}

// Sessions
export const createSession = (userId: string, context?: string) =>
  json<Session>('/sessions/', {
    method: 'POST',
    body: JSON.stringify({ user_id: userId, context }),
  })

export const endSession = (sessionId: string) =>
  json<Session>(`/sessions/${sessionId}/end`, { method: 'POST' })

// Predictions
export const listPredictions = (sessionId: string, limit = 120) =>
  json<Prediction[]>(`/predictions/?session_id=${sessionId}&limit=${limit}`)

// Health
export const getHealth = () =>
  json<{ status: string; db: string; redis: string }>('/health/ready')
