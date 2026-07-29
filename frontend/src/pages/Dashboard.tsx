import { useMemo, useState } from 'react'
import { Activity } from 'lucide-react'
import { useStream } from '@/hooks/useStream'
import { useCurrentUser } from '@/hooks/useCurrentUser'
import { useHistoricalPredictions } from '@/hooks/useHistoricalPredictions'
import { useXAISummary } from '@/hooks/useXAISummary'
import { StatusBadge } from '@/components/StatusBadge'
import { MetricGauge } from '@/components/MetricGauge'
import { EmotionBar } from '@/components/EmotionBar'
import { TimeSeriesChart } from '@/components/TimeSeriesChart'
import { XAIPanel } from '@/components/XAIPanel'
import { SessionPanel } from '@/components/dashboard/SessionPanel'
import type { MetricSeries, Prediction, Session, WsMessage } from '@/types'

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
  const { user } = useCurrentUser()
  const [activeSession, setActiveSession] = useState<Session | null>(null)
  const [paused, setPaused] = useState(false)

  // null / no session selected  -> demo mode (synthetic live stream)
  // status === 'active'         -> live WebSocket stream for the real session
  // status !== 'active'         -> completed session, read predictions over REST
  const isLive = activeSession === null || activeSession.status === 'active'
  const liveSessionId = activeSession === null ? 'demo' : activeSession.id

  const stream = useStream(isLive ? liveSessionId : null)
  const historical = useHistoricalPredictions(!isLive ? activeSession!.id : null)
  const xaiSummaryId = activeSession?.status === 'completed' ? activeSession.id : null
  const { summary: xaiSummary, loading: xaiLoading } = useXAISummary(xaiSummaryId)

  const status = isLive ? stream.status : 'closed'
  const history = isLive ? stream.history : historical.history

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
          <span className="text-xs text-gray-500 ml-2 font-mono">v0.4</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setPaused((v) => !v)}
            className="rounded-lg bg-gray-800 px-3 py-1 text-xs font-medium text-gray-300 hover:bg-gray-700 transition"
          >
            {paused ? '▶ Resume' : '⏸ Pause'}
          </button>
          {isLive ? (
            <StatusBadge status={status} />
          ) : (
            <span className="rounded-full bg-gray-800 px-3 py-1 text-xs font-medium text-gray-400">
              Historical ({historical.history.length} predictions)
            </span>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 space-y-6">

        <SessionPanel user={user} activeSession={activeSession} onSelectSession={setActiveSession} />

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
          {isLive ? (
            <XAIPanel
              weights={p?.shap_weights ?? null}
              explanation={p?.explanation_text ?? null}
            />
          ) : (
            <XAIPanel
              summary={xaiSummary}
              summaryLoading={xaiLoading}
            />
          )}
        </div>

        {/* Time-series */}
        <TimeSeriesChart data={series} />

        {/* No-data placeholder */}
        {!p && (
          <div className="rounded-xl border border-dashed border-gray-700 p-8 text-center text-sm text-gray-500">
            {isLive
              ? status === 'connecting'
                ? 'Connecting to stream…'
                : status === 'open'
                ? 'Connected — waiting for first prediction…'
                : 'Stream offline. Reconnecting automatically.'
              : historical.loading
              ? 'Loading session history…'
              : historical.error
              ? `Failed to load history: ${historical.error}`
              : 'No predictions recorded for this session.'}
          </div>
        )}
      </main>
    </div>
  )
}
