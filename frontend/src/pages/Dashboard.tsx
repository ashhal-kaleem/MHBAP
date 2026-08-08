import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Camera } from 'lucide-react'
import { useStream } from '@/hooks/useStream'
import { useHistoricalPredictions } from '@/hooks/useHistoricalPredictions'
import { useXAISummary } from '@/hooks/useXAISummary'
import { StatusBadge } from '@/components/StatusBadge'
import { MetricGauge } from '@/components/MetricGauge'
import { EmotionBar } from '@/components/EmotionBar'
import { TimeSeriesChart } from '@/components/TimeSeriesChart'
import { XAIPanel } from '@/components/XAIPanel'
import { SessionPanel } from '@/components/dashboard/SessionPanel'
import { CapturePreview } from '@/components/dashboard/CapturePreview'
import type { MetricSeries, Prediction, Session, User, WsMessage } from '@/types'

// ── helpers ───────────────────────────────────────────────────────────────────

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

// ── component ─────────────────────────────────────────────────────────────────

interface DashboardProps {
  /** Passed in from App so we don't run a second auth bootstrap here. */
  user: User | null
}

export default function Dashboard({ user }: DashboardProps) {
  const [activeSession, setActiveSession] = useState<Session | null>(null)
  const [paused, setPaused] = useState(false)
  const [capturePermErr, setCapturePermErr] = useState<string | null>(null)

  // Frozen snapshots captured the moment the user hits Pause so the UI
  // freezes in place rather than going blank.
  const frozenSeriesRef     = useRef<MetricSeries[]>([])
  const frozenPredictionRef = useRef<Prediction | null>(null)

  // A session is "live" only when it is explicitly active — no demo fallback.
  const isLive      = activeSession?.status === 'active'
  const isCompleted = activeSession?.status === 'completed'

  // Connect to WS only while a real session is active; null = no connection.
  const stream     = useStream(isLive ? activeSession!.id : null)
  const historical = useHistoricalPredictions(isCompleted ? activeSession!.id : null)
  const xaiId      = isCompleted ? activeSession!.id : null
  const { summary: xaiSummary, loading: xaiLoading } = useXAISummary(xaiId)

  const status  = isLive ? stream.status : 'closed'
  const history = isLive ? stream.history : (isCompleted ? historical.history : [])

  // ── derived state (recalculated on every history update) ──────────────────

  /** The most recent prediction frame in history, or null while waiting. */
  const livePrediction = useMemo<Prediction | null>(() => {
    if (stream.latest && isPrediction(stream.latest)) {
      return stream.latest.payload as Prediction
    }
    for (let i = history.length - 1; i >= 0; i--) {
      if (isPrediction(history[i])) return history[i].payload as Prediction
    }
    return null
  }, [history, stream.latest])

  /** Time-series array for the chart — one point per prediction frame. */
  const liveSeries = useMemo<MetricSeries[]>(() =>
    history
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
  , [history])

  // ── pause / freeze ────────────────────────────────────────────────────────

  // Snapshot synchronously during render so the UI doesn't freeze to blank.
  if (paused && !frozenPredictionRef.current) {
    frozenSeriesRef.current     = liveSeries
    frozenPredictionRef.current = livePrediction
  } else if (!paused && frozenPredictionRef.current) {
    frozenSeriesRef.current     = []
    frozenPredictionRef.current = null
  }

  const prediction = paused ? frozenPredictionRef.current : livePrediction
  const series     = paused ? frozenSeriesRef.current     : liveSeries
  const p          = prediction

  // Clear capture permission error when session ends
  useEffect(() => {
    if (!isLive) setCapturePermErr(null)
  }, [isLive])

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-ivory text-gray-900 font-sans selection:bg-sage selection:text-white">
      <header className="flex items-center justify-between border-b border-gray-200 bg-white/70 backdrop-blur-md px-6 py-4 sticky top-0 z-10">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-plum" />
          <span className="text-lg font-bold tracking-tight">MHBAP Dashboard</span>
          <span className="text-xs text-gray-500 ml-2 font-mono">v0.4</span>
        </div>
        <div className="flex items-center gap-3">
          {isLive && (
            <button
              onClick={() => setPaused((v) => !v)}
              className="rounded-lg bg-white border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 transition shadow-sm"
            >
              {paused ? '▶ Resume' : '⏸ Pause'}
            </button>
          )}
          {isLive ? (
            <StatusBadge status={status} />
          ) : isCompleted ? (
            <span className="rounded-full bg-white border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 shadow-sm">
              Historical ({historical.history.length} predictions)
            </span>
          ) : null}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 space-y-6">

        {/* Capture permission error */}
        {capturePermErr && (
          <div className="rounded-xl border border-red-300 bg-red-50 p-4 text-sm text-red-700 flex items-center gap-2">
            <span>⚠️</span>
            <span>
              <strong>Camera / microphone access denied:</strong> {capturePermErr}. Please grant
              permissions in your browser settings and click Start Analysis again.
            </span>
          </div>
        )}

        {/* Session panel: Start/Stop + history list */}
        <SessionPanel
          user={user}
          activeSession={activeSession}
          onSelectSession={setActiveSession}
        />

        {/* ── No session — idle prompt ──────────────────────────────────── */}
        {!activeSession && (
          <div className="rounded-2xl border border-dashed border-gray-200 bg-gray-50/50 p-16 flex flex-col items-center justify-center text-center min-h-[300px] gap-4">
            <Camera className="h-12 w-12 text-gray-300" />
            <div>
              <p className="text-gray-700 font-semibold text-base">Ready to start a session</p>
              <p className="text-gray-400 text-sm mt-1">
                Click <strong>Start Analysis</strong> above to begin real-time capture and
                inference. Your browser will ask for camera and microphone access.
              </p>
            </div>
          </div>
        )}

        {/* ── Active session ────────────────────────────────────────────── */}
        {isLive && (
          <>
            {/* Camera preview + mic status */}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="md:col-span-1">
                <CapturePreview
                  active={isLive}
                  onPermissionDenied={(err) => setCapturePermErr(err)}
                />
              </div>

              {/* Gauges — 2/3 width on medium+ screens */}
              <div className="md:col-span-2 rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm p-6">
                <p className="mb-4 text-sm font-semibold text-gray-900">Real-Time Signals</p>
                <div className="flex flex-wrap justify-around gap-6 h-full items-center">
                  <MetricGauge label="Stress"     value={p?.stress     ?? 0} color="stroke-red-400" />
                  <MetricGauge label="Engagement" value={p?.engagement ?? 0} color="stroke-green-400" />
                  <MetricGauge label="Attention"  value={p?.attention  ?? 0} color="stroke-blue-400" />
                  <MetricGauge label="Fatigue"    value={p?.fatigue    ?? 0} color="stroke-orange-400" />
                </div>
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

            {/* Waiting for first prediction */}
            {!p && (
              <div className="rounded-2xl border border-dashed border-gray-200 bg-gray-50/50 p-10 flex flex-col items-center justify-center text-center">
                <Activity className="h-8 w-8 text-gray-300 mb-3" />
                <p className="text-gray-600 font-medium text-sm">
                  {status === 'connecting'
                    ? 'Connecting to stream…'
                    : status === 'open'
                    ? 'Connected — waiting for first prediction…'
                    : 'Stream offline. Reconnecting automatically.'}
                </p>
              </div>
            )}
          </>
        )}

        {/* ── Completed session (historical view) ───────────────────────── */}
        {isCompleted && (
          <>
            {/* Gauges row */}
            <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm p-6">
              <p className="mb-4 text-sm font-semibold text-gray-900">Session Signals</p>
              <div className="flex flex-wrap justify-around gap-6">
                <MetricGauge label="Stress"     value={p?.stress     ?? 0} color="stroke-red-400" />
                <MetricGauge label="Engagement" value={p?.engagement ?? 0} color="stroke-green-400" />
                <MetricGauge label="Attention"  value={p?.attention  ?? 0} color="stroke-blue-400" />
                <MetricGauge label="Fatigue"    value={p?.fatigue    ?? 0} color="stroke-orange-400" />
              </div>
            </div>

            {/* Emotion + XAI summary row */}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <EmotionBar
                label={p?.emotion_label ?? '—'}
                scores={p?.emotion_scores ?? {}}
              />
              <XAIPanel
                summary={xaiSummary}
                summaryLoading={xaiLoading}
              />
            </div>

            {/* Time-series */}
            <TimeSeriesChart data={series} />

            {/* No predictions */}
            {!p && !historical.loading && (
              <div className="rounded-2xl border border-dashed border-gray-200 bg-gray-50/50 p-16 flex flex-col items-center justify-center text-center min-h-[200px]">
                <Activity className="h-8 w-8 text-gray-300 mb-3" />
                <p className="text-gray-500 text-sm">
                  {historical.error
                    ? `Failed to load history: ${historical.error}`
                    : 'No predictions recorded for this session.'}
                </p>
              </div>
            )}
            {historical.loading && (
              <div className="rounded-2xl border border-dashed border-gray-200 bg-gray-50/50 p-10 flex items-center justify-center text-gray-400 text-sm">
                Loading session history…
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}
