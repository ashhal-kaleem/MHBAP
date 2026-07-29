/** Matches backend WsMessage prediction payload */
export interface Prediction {
  id: string
  session_id: string
  /** ISO datetime — canonical field from DB schema */
  time: string
  /** Alias for time — sent by WS endpoints; absent on plain REST reads, fall back to `time` */
  recorded_at?: string
  emotion_label: string
  emotion_scores: Record<string, number>
  stress: number               // [0, 1]
  engagement: number           // [0, 1]
  attention: number            // [0, 1]
  fatigue: number              // [0, 1]
  shap_weights: Record<string, number> | null
  explanation_text: string | null
}

export interface Session {
  id: string
  user_id: string
  started_at: string
  ended_at: string | null
  status: 'active' | 'completed' | 'error'
  context: string
  consent_recording: boolean
  faces_blurred: boolean
  session_metadata: Record<string, unknown> | null
}

export interface SessionStats {
  session_id: string
  prediction_count: number
  duration_seconds: number | null
  avg_stress: number | null
  avg_engagement: number | null
  avg_attention: number | null
  avg_fatigue: number | null
  dominant_emotion: string | null
}

export interface User {
  id: string
  username: string
  email: string
  role: string
  created_at: string
}

export type WsPayload =
  | Prediction
  | Session
  | { session_id: string }
  | { message: string }
  | null

export interface WsMessage {
  type: 'prediction' | 'session_start' | 'session_end' | 'error' | 'ping'
  payload: WsPayload
}

export interface ModalityTrend {
  time: string               // ISO datetime
  weights: Record<string, number>
}

export interface XAISummary {
  session_id: string
  prediction_count: number
  avg_weights: Record<string, Record<string, number>>  // head -> modality -> avg
  trends: Record<string, ModalityTrend[]>              // head -> trend series
  dominant_modality: string | null
}

// ── Phase 10: Analytics types ─────────────────────────────────────────────

export interface MetricTimePoint {
  time: string    // ISO datetime
  value: number
}

export interface MetricTimeSeries {
  metric: string
  points: MetricTimePoint[]
}

export interface EmotionBreakdown {
  counts: Record<string, number>
  total: number
}

export interface SessionSummary {
  session_id: string
  context: string | null
  started_at: string
  ended_at: string | null
  duration_seconds: number | null
  prediction_count: number
  avg_stress: number | null
  avg_engagement: number | null
  avg_attention: number | null
  avg_fatigue: number | null
  dominant_emotion: string | null
}

export interface UserAnalytics {
  user_id: string
  session_count: number
  total_duration_seconds: number
  sessions: SessionSummary[]
  metric_trends: Record<string, MetricTimeSeries>
  emotion_breakdown: EmotionBreakdown
}

export type ConnectionStatus = 'connecting' | 'open' | 'closed' | 'error'

export interface MetricSeries {
  time: string     // HH:mm:ss display label
  stress: number
  engagement: number
  attention: number
  fatigue: number
}
