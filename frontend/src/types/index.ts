/** Matches backend WsMessage prediction payload */
export interface Prediction {
  id: string
  session_id: string
  /** ISO datetime — canonical field from DB schema */
  time: string
  /** Alias for time — sent by both /demo and /session endpoints */
  recorded_at: string
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
  context: string | null
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

export type ConnectionStatus = 'connecting' | 'open' | 'closed' | 'error'

export interface MetricSeries {
  time: string     // HH:mm:ss display label
  stress: number
  engagement: number
  attention: number
  fatigue: number
}
