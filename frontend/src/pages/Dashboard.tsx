import { useMemo, useState } from 'react'
import { Activity } from 'lucide-react'
import { useStream } from '@/hooks/useStream'
import { StatusBadge } from '@/components/StatusBadge'
import { MetricGauge } from '@/components/MetricGauge'
import { EmotionBar } from '@/components/EmotionBar'
import { TimeSeriesChart } from '@/components/TimeSeriesChart'
import { XAIPanel } from '@/components/XAIPanel'
import type { MetricSeries, Prediction, WsMessage } from '@/types'

/**
 * Session to connect to.
 *   "demo"   → synthetic stream, no hardware needed
 *   <uuid>   → live session stream
 * Override via VITE_DEMO_SESSION_ID env var.
 */
const SESSION_ID: string =
  (import.meta.env.VITE_DEMO_SESSION_ID as string | undefined) ?? 'demo'

function isPrediction(msg: WsMessage): msg is WsMessage & { payload: Prediction } {
  return msg.type === 'prediction' && msg.payload !== null
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('en-US', { hour12: false })
  } catch {
    return iso
  }
}

export default function Dashboard() {
  const { status, history } = useStream(SESSION_ID)
  const [paused, setPaused] = useState(false)

  const prediction = useMemo<Prediction | null>(() => {
    for (let i = history.length - 1; i >= 0; i--) {
      if (isPrediction(history[i])) return history[i].payload as Prediction
    }
    return null
  }, [history])

  const series = useMemo<MetricSeries[]>(() => {
    if (paused) return []
    return history
      .filter(isPrediction)
      .map((m) => {
        const p = m.payload as Prediction
        return {
          time:       formatTime(p.recorded_at ?? p.time),
          stress:     p.stress,
          engagement: p.engagement,
          attention:  p.attention,
          fatigue:    p.fatigue,
        }
      })
  }, [history, paused])

  const p = prediction

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans">
      <header className="flex items-center justify-between border-b border-gray-800 px-6 py-4">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-blue-500" />
          <span className="text-lg font-bold tracking-tight">MHBAP Dashboard</span>
          <span className="text-xs text-gray-500 ml-2 font-mono">v0.3</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setPaused((v) => !v)}
            className="rounded-lg bg-gray-800 px-3 py-1 text-xs font-medium text-gray-300 hover:bg-gray-700 transition"
          >
            {paused ? '▶ Resume' : '⏸ Pause'}
          </button>
          <StatusBadge status={status} />
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 space-y-6">

        {/* Gauges row */}
        <div className="rounded-xl bg-gray-800 p-5">
          <p className="mb-4 text-sm font-semibold text-gray-200">Real-Time Signals</p>
          <div className="flex flex-wrap justify-around gap-6">
            <MetricGauge label="Stress"     value={p?.stress     ?? 0} color="stroke-red-400" />
            <MetricGauge label="Engagement" value={p?.engagement ?? 0} color="stroke-green-400" />
            <MetricGauge label="Attention"  value={p?.attention  ?? 0} color="stroke-blue-400" />
            <MetricGauge label="Fatigue"    value={p?.fatigue    ?? 0} color="stroke-orange-400" />
          </div>
        </div>

        {/* Emotion + XAI row */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <EmotionBar
            label={p?.emotion_label ?? '—'}
            scores={p?.emotion_scores ?? {}}
          />
          <XAIPanel
            weights={p?.shap_weights ?? null}
            explanation={p?.explanation_text ?? null}
          />
        </div>

        {/* Time-series */}
        <TimeSeriesChart data={series} />

        {/* No-data placeholder */}
        {!p && (
          <div className="rounded-xl border border-dashed border-gray-700 p-8 text-center text-sm text-gray-500">
            {status === 'connecting'
              ? 'Connecting to stream…'
              : status === 'open'
              ? 'Connected — waiting for first prediction…'
              : 'Stream offline. Reconnecting automatically.'}
          </div>
        )}
      </main>
    </div>
  )
}
