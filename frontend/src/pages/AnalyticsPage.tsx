/**
 * AnalyticsPage — cross-session analytics dashboard (Phase 10).
 *
 * Displays:
 *  - Summary cards (total sessions, total duration, dominant emotion)
 *  - Metric trend chart (stress/engagement/attention/fatigue over sessions)
 *  - Emotion breakdown bar chart
 *  - Session table with per-row JSON/CSV export
 *  - Bulk CSV export for all sessions
 */
import { useState } from 'react'
import {
  AreaChart, Area, BarChart, Bar,
  CartesianGrid, Legend, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { exportSessionCsv, exportSessionJson, exportUserCsv } from '@/services/api'
import { useCurrentUser } from '@/hooks/useCurrentUser'
import { useUserAnalytics } from '@/hooks/useUserAnalytics'
import type { SessionSummary } from '@/types'

// Colour palette (matches dashboard gauges)
const METRIC_COLORS = {
  stress:     '#ef4444',
  engagement: '#22c55e',
  attention:  '#3b82f6',
  fatigue:    '#f59e0b',
}

const EMOTION_COLOR = '#8b5cf6'

function fmtDur(secs: number | null): string {
  if (secs == null) return '—'
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

function pct(v: number | null): string {
  return v != null ? `${Math.round(v * 100)}%` : '—'
}

// ── Sub-components ────────────────────────────────────────────────────────

function SummaryCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-800 rounded-xl p-4 flex flex-col gap-1">
      <span className="text-xs text-gray-400 uppercase tracking-widest">{label}</span>
      <span className="text-2xl font-bold text-white">{value}</span>
    </div>
  )
}

function MetricTrendChart({ trends }: { trends: Record<string, { metric: string; points: { time: string; value: number }[] }> }) {
  // Merge all metrics into [{label, stress, engagement, attention, fatigue}]
  const labels = trends['stress']?.points.map((_, i) => `S${i + 1}`) ?? []
  const data = labels.map((label, i) => ({
    label,
    stress:     trends['stress']?.points[i]?.value ?? 0,
    engagement: trends['engagement']?.points[i]?.value ?? 0,
    attention:  trends['attention']?.points[i]?.value ?? 0,
    fatigue:    trends['fatigue']?.points[i]?.value ?? 0,
  }))

  if (data.length === 0) {
    return <p className="text-gray-500 text-sm text-center py-8">No trend data yet.</p>
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="label" tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <Tooltip formatter={(v: number) => `${Math.round(v * 100)}%`} contentStyle={{ background: '#1f2937', border: 'none' }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {(Object.entries(METRIC_COLORS) as [string, string][]).map(([key, color]) => (
          <Area key={key} type="monotone" dataKey={key} stroke={color} fill={color} fillOpacity={0.15} strokeWidth={2} dot={{ r: 3 }} />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  )
}

function EmotionChart({ breakdown }: { breakdown: { counts: Record<string, number>; total: number } }) {
  const data = Object.entries(breakdown.counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name, count, pct: Math.round((count / breakdown.total) * 100) }))

  if (data.length === 0) {
    return <p className="text-gray-500 text-sm text-center py-8">No emotion data yet.</p>
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ left: 16, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
        <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <YAxis type="category" dataKey="name" width={90} tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <Tooltip formatter={(v: number) => [`${v} predictions`]} contentStyle={{ background: '#1f2937', border: 'none' }} />
        <Bar dataKey="count" fill={EMOTION_COLOR} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
function SessionTable({ sessions }: { sessions: SessionSummary[] }) {
  const [exporting, setExporting] = useState<string | null>(null)

  async function doExport(sid: string, fmt: 'csv' | 'json') {
    setExporting(`${sid}-${fmt}`)
    try {
      if (fmt === 'csv') await exportSessionCsv(sid)
      else await exportSessionJson(sid)
    } finally {
      setExporting(null)
    }
  }

  if (sessions.length === 0) {
    return <p className="text-gray-500 text-sm text-center py-8">No sessions recorded yet.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead className="text-xs text-gray-400 uppercase border-b border-gray-700">
          <tr>
            <th className="py-2 pr-4">Date</th>
            <th className="py-2 pr-4">Context</th>
            <th className="py-2 pr-4">Duration</th>
            <th className="py-2 pr-4">Predictions</th>
            <th className="py-2 pr-4">Stress</th>
            <th className="py-2 pr-4">Engagement</th>
            <th className="py-2 pr-4">Emotion</th>
            <th className="py-2">Export</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {sessions.map((s) => (
            <tr key={s.session_id} className="hover:bg-gray-800/50 transition-colors">
              <td className="py-2 pr-4 text-gray-300">{fmtDate(s.started_at)}</td>
              <td className="py-2 pr-4 text-gray-300 max-w-[120px] truncate">{s.context ?? '—'}</td>
              <td className="py-2 pr-4 text-gray-400">{fmtDur(s.duration_seconds)}</td>
              <td className="py-2 pr-4 text-gray-400">{s.prediction_count}</td>
              <td className="py-2 pr-4">
                <span className={s.avg_stress != null && s.avg_stress > 0.65 ? 'text-red-400' : 'text-gray-300'}>
                  {pct(s.avg_stress)}
                </span>
              </td>
              <td className="py-2 pr-4">
                <span className={s.avg_engagement != null && s.avg_engagement > 0.6 ? 'text-green-400' : 'text-gray-300'}>
                  {pct(s.avg_engagement)}
                </span>
              </td>
              <td className="py-2 pr-4">
                <span className="capitalize text-purple-400">{s.dominant_emotion ?? '—'}</span>
              </td>
              <td className="py-2 flex gap-2">
                <button
                  onClick={() => doExport(s.session_id, 'csv')}
                  disabled={exporting === `${s.session_id}-csv`}
                  className="text-xs px-2 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 disabled:opacity-50"
                >CSV</button>
                <button
                  onClick={() => doExport(s.session_id, 'json')}
                  disabled={exporting === `${s.session_id}-json`}
                  className="text-xs px-2 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 disabled:opacity-50"
                >JSON</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
// ── Main page ─────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const { user } = useCurrentUser()
  const userId = user?.id ?? null
  const { analytics, loading, error, refresh } = useUserAnalytics(userId)
  const [exporting, setExporting] = useState(false)

  async function handleBulkExport() {
    if (!userId) return
    setExporting(true)
    try { await exportUserCsv(userId) } finally { setExporting(false) }
  }

  if (loading && !analytics) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
        Loading analytics…
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col items-center gap-4 h-64 justify-center">
        <p className="text-red-400 text-sm">{error}</p>
        <button onClick={refresh} className="text-xs px-3 py-1 rounded bg-gray-700 text-gray-200 hover:bg-gray-600">
          Retry
        </button>
      </div>
    )
  }

  if (!analytics) return null

  const totalDurMin = Math.round(analytics.total_duration_seconds / 60)

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Behaviour Analytics</h1>
          <p className="text-sm text-gray-400 mt-1">Cross-session insights for your recording history</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={refresh}
            className="text-xs px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-200"
          >↻ Refresh</button>
          <button
            onClick={handleBulkExport}
            disabled={exporting || analytics.session_count === 0}
            className="text-xs px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50"
          >{exporting ? 'Exporting…' : '⬇ Export All CSV'}</button>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <SummaryCard label="Sessions" value={analytics.session_count} />
        <SummaryCard label="Total Time" value={`${totalDurMin} min`} />
        <SummaryCard
          label="Total Predictions"
          value={analytics.sessions.reduce((s, x) => s + x.prediction_count, 0)}
        />
        <SummaryCard
          label="Dominant Emotion"
          value={analytics.sessions.length > 0
            ? (Object.entries(analytics.emotion_breakdown.counts)
                .sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—')
            : '—'}
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <div className="lg:col-span-3 bg-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Metric Trends Across Sessions</h2>
          <MetricTrendChart trends={analytics.metric_trends} />
        </div>
        <div className="lg:col-span-2 bg-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Emotion Breakdown</h2>
          <EmotionChart breakdown={analytics.emotion_breakdown} />
        </div>
      </div>

      {/* Session table */}
      <div className="bg-gray-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Session History</h2>
        <SessionTable sessions={analytics.sessions} />
      </div>
    </div>
  )
}
