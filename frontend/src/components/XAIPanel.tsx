/**
 * XAIPanel — Explainable AI panel with two views:
 *
 *  LIVE view   (weights prop): shows real-time SHAP bars + NL explanation
 *              with per-head tab selector (stress / engagement / attention / fatigue).
 *
 *  SUMMARY view (summary prop): shows session-level averaged SHAP contributions
 *              with dominant modality highlight and per-head tab selector.
 *
 * Usage:
 *   <XAIPanel weights={p?.shap_weights} explanation={p?.explanation_text} />
 *   <XAIPanel summary={xaiSummary} summaryLoading={loading} />
 */
import { useState } from 'react'
import { clsx } from 'clsx'
import type { XAISummary } from '@/types'

const HEADS = ['stress', 'engagement', 'attention', 'fatigue'] as const
type Head = typeof HEADS[number]

const HEAD_COLOR: Record<Head, string> = {
  stress:     'text-red-400 border-red-400',
  engagement: 'text-green-400 border-green-400',
  attention:  'text-blue-400 border-blue-400',
  fatigue:    'text-orange-400 border-orange-400',
}

const MODAL_BAR: Record<string, string> = {
  face:  'bg-blue-500',
  voice: 'bg-green-500',
  gaze:  'bg-purple-500',
  pose:  'bg-yellow-500',
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
  // Live mode
  weights?: Record<string, number> | null
  explanation?: string | null
  // Summary mode
  summary?: XAISummary | null
  summaryLoading?: boolean
}

function ContribBars({ weights }: { weights: Record<string, number> }) {
  if (!weights || Object.keys(weights).length === 0) {
    return <p className="text-xs text-gray-500">No modality data.</p>
  }
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1])
  return (
    <div className="space-y-2">
      {entries.map(([mod, w]) => (
        <div key={mod} className="space-y-0.5">
          <div className="flex justify-between text-xs text-gray-400">
            <span>{MODAL_LABELS[mod] ?? mod}</span>
            <span className="font-mono">{Math.round(w * 100)}%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-gray-700 overflow-hidden">
            <div
              className={clsx('h-2 rounded-full transition-all duration-500', MODAL_BAR[mod] ?? 'bg-gray-500')}
              style={{ width: `${Math.min(w * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

export function XAIPanel({ weights, explanation, summary, summaryLoading }: Props) {
  const [activeHead, setActiveHead] = useState<Head>('stress')

  const isSummaryMode = summary != null || summaryLoading

  // ── empty state ──────────────────────────────────────────────────────────
  if (!weights && !isSummaryMode) {
    return (
      <div className="rounded-xl bg-gray-800 p-4 text-sm text-gray-500">
        XAI data will appear once the inference pipeline is running.
      </div>
    )
  }

  // ── head tabs ─────────────────────────────────────────────────────────────
  const TabBar = () => (
    <div className="flex gap-1 mb-3">
      {HEADS.map((h) => (
        <button
          key={h}
          onClick={() => setActiveHead(h)}
          className={clsx(
            'rounded-md px-2 py-1 text-[10px] font-medium capitalize border transition',
            activeHead === h
              ? HEAD_COLOR[h]
              : 'border-transparent text-gray-500 hover:text-gray-300',
          )}
        >
          {h}
        </button>
      ))}
    </div>
  )

  // ── SUMMARY mode ─────────────────────────────────────────────────────────
  if (isSummaryMode) {
    const headWeights = summary?.avg_weights?.[activeHead] ?? {}
    return (
      <div className="rounded-xl bg-gray-800 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-gray-200">SHAP — Session Average</p>
          {summary && (
            <span className="text-[10px] text-gray-500">
              {summary.prediction_count} predictions
            </span>
          )}
        </div>

        {summaryLoading && <p className="text-xs text-gray-500">Computing XAI summary…</p>}

        {summary && (
          <>
            {summary.dominant_modality && (
              <p className="text-xs text-gray-400">
                Dominant signal:{' '}
                <span className="font-medium text-white capitalize">
                  {MODAL_LABELS[summary.dominant_modality] ?? summary.dominant_modality}
                </span>
              </p>
            )}
            <TabBar />
            <ContribBars weights={headWeights} />
          </>
        )}
      </div>
    )
  }

  // ── LIVE mode ─────────────────────────────────────────────────────────────
  const liveWeights = weights ?? {}
  return (
    <div className="rounded-xl bg-gray-800 p-4 space-y-3">
      <p className="text-sm font-semibold text-gray-200">Modality Contributions (SHAP)</p>
      <TabBar />
      <ContribBars weights={liveWeights} />
      {explanation && (
        <p className="mt-2 rounded-lg bg-gray-700/60 px-3 py-2 text-xs text-gray-300 leading-relaxed border border-gray-700">
          {explanation}
        </p>
      )}
    </div>
  )
}
