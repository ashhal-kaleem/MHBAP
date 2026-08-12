/**
 * XAIPanel — Explainable AI panel with two views:
 *
 *  LIVE view   (weights prop): shows real-time SHAP bars + NL explanation
 *  SUMMARY view (summary prop): shows session-level averaged SHAP contributions
 */
import { useState } from 'react'
import { clsx } from 'clsx'
import { Sparkles } from 'lucide-react'
import type { XAISummary } from '@/types'

const HEADS = ['stress', 'engagement', 'attention', 'fatigue'] as const
type Head = typeof HEADS[number]

const HEAD_STYLE: Record<Head, { active: string; dot: string }> = {
  stress:     { active: 'bg-red-100 text-red-700 border-red-200',    dot: 'bg-red-400' },
  engagement: { active: 'bg-green-100 text-green-700 border-green-200', dot: 'bg-green-400' },
  attention:  { active: 'bg-blue-100 text-blue-700 border-blue-200',   dot: 'bg-blue-400' },
  fatigue:    { active: 'bg-orange-100 text-orange-700 border-orange-200', dot: 'bg-orange-400' },
}

const MODAL_BAR: Record<string, string> = {
  face:  'bg-blue-500',
  voice: 'bg-emerald-500',
  gaze:  'bg-violet-500',
  pose:  'bg-amber-500',
  hci:   'bg-pink-500',
}

const MODAL_LABELS: Record<string, string> = {
  face:  'Face',
  voice: 'Voice',
  gaze:  'Gaze',
  pose:  'Posture',
  hci:   'HCI',
}

interface Props {
  weights?: Record<string, number> | null
  explanation?: string | null
  summary?: XAISummary | null
  summaryLoading?: boolean
}

function ContribBars({ weights, waiting = false }: { weights: Record<string, number>; waiting?: boolean }) {
  if (waiting) {
    return (
      <div className="space-y-3">
        {[70, 45, 30, 20, 10].map((w) => (
          <div key={w} className="space-y-1">
            <div className="flex justify-between">
              <div className="h-3 w-14 rounded bg-gray-100 animate-pulse" />
              <div className="h-3 w-8 rounded bg-gray-100 animate-pulse" />
            </div>
            <div className="h-3 w-full rounded-full bg-gray-100 overflow-hidden">
              <div className="h-3 rounded-full bg-gray-200 animate-pulse" style={{ width: `${w}%` }} />
            </div>
          </div>
        ))}
      </div>
    )
  }

  if (!weights || Object.keys(weights).length === 0) {
    return <p className="text-xs text-gray-400 text-center py-4">No modality data available.</p>
  }

  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1])
  return (
    <div className="space-y-3">
      {entries.map(([mod, w]) => (
        <div key={mod} className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-gray-700 font-medium">{MODAL_LABELS[mod] ?? mod}</span>
            <span className="font-mono text-gray-500 tabular-nums">{Math.round(w * 100)}%</span>
          </div>
          <div className="h-3 w-full rounded-full bg-gray-100 overflow-hidden">
            <div
              className={clsx('h-3 rounded-full transition-all duration-500', MODAL_BAR[mod] ?? 'bg-gray-400')}
              style={{ width: `${Math.min(w * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

function TabBar({ activeHead, setActiveHead }: { activeHead: Head; setActiveHead: (h: Head) => void }) {
  return (
    <div className="flex gap-1.5 flex-wrap">
      {HEADS.map((h) => {
        const style = HEAD_STYLE[h]
        const isActive = activeHead === h
        return (
          <button
            key={h}
            onClick={() => setActiveHead(h)}
            className={clsx(
              'flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-semibold border capitalize transition-all',
              isActive
                ? style.active
                : 'border-transparent text-gray-400 hover:text-gray-700 hover:bg-gray-50',
            )}
          >
            {isActive && <span className={clsx('w-1.5 h-1.5 rounded-full', style.dot)} />}
            {h}
          </button>
        )
      })}
    </div>
  )
}

export function XAIPanel({ weights, explanation, summary, summaryLoading }: Props) {
  const [activeHead, setActiveHead] = useState<Head>('stress')
  const isSummaryMode = summary != null || summaryLoading

  if (!weights && !isSummaryMode) {
    return (
      <div className="rounded-2xl bg-white border border-gray-100 shadow-sm p-6 flex flex-col items-center justify-center text-center gap-3 min-h-[220px]">
        <div className="w-12 h-12 rounded-xl bg-gray-50 flex items-center justify-center">
          <Sparkles className="w-6 h-6 text-gray-300" />
        </div>
        <p className="text-sm text-gray-400 max-w-[200px]">
          XAI contributions will appear once the inference pipeline is running.
        </p>
      </div>
    )
  }

  if (isSummaryMode) {
    const headWeights = summary?.avg_weights?.[activeHead] ?? {}
    return (
      <div className="rounded-2xl bg-white border border-gray-100 shadow-sm p-5 space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-gray-900">SHAP — Session Average</p>
          {summary && (
            <span className="text-[11px] text-gray-400 font-mono">
              {summary.prediction_count} frames
            </span>
          )}
        </div>
        {summary?.dominant_modality && (
          <p className="text-xs text-gray-500">
            Dominant signal:{' '}
            <span className="font-semibold text-gray-800 capitalize">
              {MODAL_LABELS[summary.dominant_modality] ?? summary.dominant_modality}
            </span>
          </p>
        )}
        <TabBar activeHead={activeHead} setActiveHead={setActiveHead} />
        <ContribBars weights={headWeights} waiting={summaryLoading && !summary} />
      </div>
    )
  }

  return (
    <div className="rounded-2xl bg-white border border-gray-100 shadow-sm p-5 space-y-4">
      <p className="text-sm font-semibold text-gray-900">Modality Contributions (SHAP)</p>
      <TabBar activeHead={activeHead} setActiveHead={setActiveHead} />
      <ContribBars weights={weights ?? {}} />
      {explanation && (
        <div className="rounded-xl bg-plum/5 border border-plum/10 px-4 py-3 flex gap-2">
          <Sparkles className="h-4 w-4 text-plum shrink-0 mt-0.5" />
          <p className="text-xs text-plum-dark leading-relaxed">{explanation}</p>
        </div>
      )}
    </div>
  )
}
