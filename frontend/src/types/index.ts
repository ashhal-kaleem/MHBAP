/** Matches backend app/schemas/prediction.py */
export interface Prediction {
  id: string
  session_id: string
  recorded_at: string          // ISO datetime
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

export interface WsMessage {
  type: 'prediction' | 'session_start' | 'session_end' | 'error' | 'ping'
  payload: Prediction | Session | { message: string } | null
}

export type ConnectionStatus = 'connecting' | 'open' | 'closed' | 'error'

export interface MetricSeries {
  time: string     // HH:mm:ss
  stress: number
  engagement: number
  attention: number
  fatigue: number
}
