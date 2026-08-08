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
          <div className="flex justify-between text-xs text-gray-500">
            <span>{MODAL_LABELS[mod] ?? mod}</span>
            <span className="font-mono text-gray-900">{Math.round(w * 100)}%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-gray-100 overflow-hidden">
            <div
              className={clsx('h-2 rounded-full transition-all duration-500', MODAL_BAR[mod] ?? 'bg-gray-400')}
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
      <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm p-8 flex flex-col items-center justify-center text-center text-sm text-gray-500 min-h-[250px]">
        <svg className="w-8 h-8 mx-auto mb-3 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <p>XAI data will appear once the inference pipeline is running.</p>
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
              : 'border-transparent text-gray-400 hover:text-gray-600 hover:bg-gray-50',
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
      <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm p-5 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-gray-900">SHAP — Session Average</p>
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
              <p className="text-xs text-gray-500">
                Dominant signal:{' '}
                <span className="font-medium text-gray-900 capitalize">
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
    <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm p-5 space-y-3">
      <p className="text-sm font-semibold text-gray-900">Modality Contributions (SHAP)</p>
      <TabBar />
      <ContribBars weights={liveWeights} />
      {explanation && (
        <p className="mt-2 rounded-xl bg-blue-50/80 px-4 py-3 text-xs text-blue-900 leading-relaxed border border-blue-100">
          {explanation}
        </p>
      )}
    </div>
  )
}
