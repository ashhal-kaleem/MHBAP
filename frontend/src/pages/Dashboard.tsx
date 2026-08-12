import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Camera, Pause, Play, Wifi, WifiOff } from 'lucide-react'
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

function isPrediction(msg: WsMessage): msg is WsMessage & { payload: Prediction } {
  return msg.type === 'prediction' && msg.payload !== null
}

function formatTime(iso: string): string {
  try { return new Date(iso).toLocaleTimeString('en-US', { hour12: false }) }
  catch { return iso }
}

interface DashboardProps {
  user: User | null
}

export default function Dashboard({ user }: DashboardProps) {
  const [activeSession, setActiveSession] = useState<Session | null>(null)
  const [paused, setPaused] = useState(false)
  const [capturePermErr, setCapturePermErr] = useState<string | null>(null)

  const frozenSeriesRef     = useRef<MetricSeries[]>([])
  const frozenPredictionRef = useRef<Prediction | null>(null)

  const isLive      = activeSession?.status === 'active'
  const isCompleted = activeSession?.status === 'completed'

  const stream     = useStream(isLive ? activeSession!.id : null)
  const historical = useHistoricalPredictions(isCompleted ? activeSession!.id : null)
  const xaiId      = isCompleted ? activeSession!.id : null
  const { summary: xaiSummary, loading: xaiLoading } = useXAISummary(xaiId)

  const status  = isLive ? stream.status : 'closed'
  const history = isLive ? stream.history : (isCompleted ? historical.history : [])

  const livePrediction = useMemo<Prediction | null>(() => {
    if (stream.latest && isPrediction(stream.latest)) return stream.latest.payload as Prediction
    for (let i = history.length - 1; i >= 0; i--) {
      if (isPrediction(history[i])) return history[i].payload as Prediction
    }
    return null
  }, [history, stream.latest])

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

  useEffect(() => { if (!isLive) setCapturePermErr(null) }, [isLive])

  return (
    <div className="min-h-screen bg-ivory text-gray-900 font-sans">
      {/* ── Sticky sub-header (session status bar) ── */}
      {(isLive || isCompleted) && (
        <div className="sticky top-14 z-40 bg-white/70 backdrop-blur-sm border-b border-gray-100 px-4 sm:px-6 py-2 flex items-center gap-3">
          <Activity className="h-4 w-4 text-plum shrink-0" />
          <span className="text-xs font-semibold text-gray-700">
            {isLive ? 'Live Session' : `Reviewing: ${activeSession?.context ?? 'Session'}`}
          </span>
          <div className="flex items-center gap-2 ml-auto">
            {isLive && (
              <button
                onClick={() => setPaused((v) => !v)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 transition shadow-sm"
              >
                {paused ? <><Play className="h-3 w-3" /> Resume</> : <><Pause className="h-3 w-3" /> Pause</>}
              </button>
            )}
            {isLive ? (
              <StatusBadge status={status} />
            ) : isCompleted ? (
              <span className="rounded-full bg-white border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 shadow-sm">
                {historical.history.length} predictions
              </span>
            ) : null}
          </div>
        </div>
      )}

      <main className="mx-auto max-w-7xl px-4 sm:px-6 py-6 space-y-5">
        {/* Capture error */}
        {capturePermErr && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 flex items-start gap-2">
            <WifiOff className="h-4 w-4 mt-0.5 shrink-0" />
            <span>
              <strong>Camera / microphone access denied:</strong> {capturePermErr}. Please grant
              permissions in your browser settings and restart the session.
            </span>
          </div>
        )}

        {/* Session control panel */}
        <SessionPanel
          user={user}
          activeSession={activeSession}
          onSelectSession={setActiveSession}
        />

        {/* ── Idle state ── */}
        {!activeSession && (
          <div className="rounded-2xl border-2 border-dashed border-gray-200 bg-white/40 p-14 flex flex-col items-center justify-center text-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-plum/8 flex items-center justify-center">
              <Camera className="h-8 w-8 text-plum/60" />
            </div>
            <div>
              <p className="text-gray-800 font-semibold">Ready when you are</p>
              <p className="text-gray-400 text-sm mt-1 max-w-sm">
                Click <strong>Start Analysis</strong> in the Session panel above. Your camera and
                microphone will be accessed by the backend pipeline.
              </p>
            </div>
          </div>
        )}

        {/* ── Live session ── */}
        {isLive && (
          <>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="md:col-span-1">
                <CapturePreview
                  active={isLive}
                  frameCount={history.filter(isPrediction).length}
                  prediction={livePrediction}
                  onPermissionDenied={(err) => setCapturePermErr(err)}
                />
              </div>
              <div className="md:col-span-2 rounded-2xl bg-white border border-gray-100 shadow-sm p-6">
                <div className="flex items-center justify-between mb-5">
                  <p className="text-sm font-semibold text-gray-900">Real-Time Signals</p>
                  {status === 'connecting' && (
                    <span className="flex items-center gap-1.5 text-xs text-amber-600 font-medium">
                      <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
                      Connecting…
                    </span>
                  )}
                  {status === 'open' && !p && (
                    <span className="flex items-center gap-1.5 text-xs text-blue-600 font-medium">
                      <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse" />
                      Waiting for first frame…
                    </span>
                  )}
                  {status === 'error' && (
                    <span className="flex items-center gap-1.5 text-xs text-red-500 font-medium">
                      <Wifi className="h-3 w-3" /> Stream error — retrying
                    </span>
                  )}
                </div>
                <div className="flex flex-wrap justify-around gap-6 items-center">
                  <MetricGauge label="Stress"     value={p?.stress     ?? 0} color="stroke-red-400"    waiting={!p} />
                  <MetricGauge label="Engagement" value={p?.engagement ?? 0} color="stroke-green-400"  waiting={!p} />
                  <MetricGauge label="Attention"  value={p?.attention  ?? 0} color="stroke-blue-400"   waiting={!p} />
                  <MetricGauge label="Fatigue"    value={p?.fatigue    ?? 0} color="stroke-orange-400" waiting={!p} />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <EmotionBar label={p?.emotion_label ?? '—'} scores={p?.emotion_scores ?? {}} />
              <XAIPanel weights={p?.shap_weights ?? null} explanation={p?.explanation_text ?? null} />
            </div>

            <TimeSeriesChart data={series} />
          </>
        )}

        {/* ── Completed session ── */}
        {isCompleted && (
          <>
            <div className="rounded-2xl bg-white border border-gray-100 shadow-sm p-6">
              <p className="mb-5 text-sm font-semibold text-gray-900">Session Signals</p>
              <div className="flex flex-wrap justify-around gap-6">
                <MetricGauge label="Stress"     value={p?.stress     ?? 0} color="stroke-red-400" />
                <MetricGauge label="Engagement" value={p?.engagement ?? 0} color="stroke-green-400" />
                <MetricGauge label="Attention"  value={p?.attention  ?? 0} color="stroke-blue-400" />
                <MetricGauge label="Fatigue"    value={p?.fatigue    ?? 0} color="stroke-orange-400" />
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <EmotionBar label={p?.emotion_label ?? '—'} scores={p?.emotion_scores ?? {}} />
              <XAIPanel summary={xaiSummary} summaryLoading={xaiLoading} />
            </div>

            <TimeSeriesChart data={series} />

            {!p && !historical.loading && (
              <div className="rounded-2xl border border-dashed border-gray-200 bg-white/40 p-14 flex flex-col items-center justify-center text-center gap-3">
                <Activity className="h-8 w-8 text-gray-300" />
                <p className="text-gray-500 text-sm">
                  {historical.error
                    ? `Failed to load history: ${historical.error}`
                    : 'No predictions recorded for this session.'}
                </p>
              </div>
            )}
            {historical.loading && (
              <div className="rounded-2xl border border-gray-100 bg-white/60 p-10 flex items-center justify-center gap-3 text-gray-400 text-sm">
                <div className="w-4 h-4 border-2 border-plum/30 border-t-plum rounded-full animate-spin" />
                Loading session history…
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}
