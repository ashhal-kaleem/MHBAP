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
    <div className="bg-white border border-gray-100 shadow-sm rounded-2xl p-5 flex flex-col gap-1">
      <span className="text-xs text-gray-500 uppercase tracking-widest font-semibold">{label}</span>
      <span className="text-3xl font-bold text-gray-900">{value}</span>
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
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="label" tick={{ fill: '#6b7280', fontSize: 11 }} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fill: '#6b7280', fontSize: 11 }} />
        <Tooltip formatter={(v: number) => `${Math.round(v * 100)}%`} contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #f3f4f6', borderRadius: '0.75rem', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', color: '#374151' }} itemStyle={{ color: '#4b5563', fontSize: '0.875rem' }} />
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
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
        <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 11 }} />
        <YAxis type="category" dataKey="name" width={90} tick={{ fill: '#6b7280', fontSize: 11 }} />
        <Tooltip formatter={(v: number) => [`${v} predictions`]} contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #f3f4f6', borderRadius: '0.75rem', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', color: '#374151' }} itemStyle={{ color: '#4b5563', fontSize: '0.875rem' }} />
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
        <thead className="text-xs text-gray-500 uppercase border-b border-gray-200">
          <tr>
            <th className="py-3 pr-4">Date</th>
            <th className="py-3 pr-4">Context</th>
            <th className="py-3 pr-4">Duration</th>
            <th className="py-3 pr-4">Predictions</th>
            <th className="py-3 pr-4">Stress</th>
            <th className="py-3 pr-4">Engagement</th>
            <th className="py-3 pr-4">Emotion</th>
            <th className="py-3">Export</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {sessions.map((s) => (
            <tr key={s.session_id} className="hover:bg-gray-50/80 transition-colors">
              <td className="py-3 pr-4 text-gray-700">{fmtDate(s.started_at)}</td>
              <td className="py-3 pr-4 text-gray-700 max-w-[120px] truncate">{s.context ?? '—'}</td>
              <td className="py-3 pr-4 text-gray-500">{fmtDur(s.duration_seconds)}</td>
              <td className="py-3 pr-4 text-gray-500">{s.prediction_count}</td>
              <td className="py-3 pr-4">
                <span className={s.avg_stress != null && s.avg_stress > 0.65 ? 'text-red-600 font-medium' : 'text-gray-700'}>
                  {pct(s.avg_stress)}
                </span>
              </td>
              <td className="py-3 pr-4">
                <span className={s.avg_engagement != null && s.avg_engagement > 0.6 ? 'text-green-600 font-medium' : 'text-gray-700'}>
                  {pct(s.avg_engagement)}
                </span>
              </td>
              <td className="py-3 pr-4">
                <span className="capitalize text-plum font-medium">{s.dominant_emotion ?? '—'}</span>
              </td>
              <td className="py-3 flex gap-2">
                <button
                  onClick={() => doExport(s.session_id, 'csv')}
                  disabled={exporting === `${s.session_id}-csv`}
                  className="text-xs px-2 py-1 rounded bg-white border border-gray-200 hover:bg-gray-50 text-gray-700 disabled:opacity-50 shadow-sm transition"
                >CSV</button>
                <button
                  onClick={() => doExport(s.session_id, 'json')}
                  disabled={exporting === `${s.session_id}-json`}
                  className="text-xs px-2 py-1 rounded bg-white border border-gray-200 hover:bg-gray-50 text-gray-700 disabled:opacity-50 shadow-sm transition"
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
      <div className="min-h-screen bg-ivory flex items-center justify-center font-sans">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-4 border-plum/30 border-t-plum rounded-full animate-spin" />
          <p className="text-gray-500 font-medium">Loading analytics…</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-ivory flex flex-col items-center justify-center font-sans">
        <div className="bg-red-50 text-red-700 p-6 rounded-2xl max-w-md text-center border border-red-100 shadow-sm">
          <p className="font-semibold mb-4">{error}</p>
          <button onClick={refresh} className="px-4 py-2 rounded-xl bg-red-600 text-white hover:bg-red-700 font-medium shadow-sm transition">
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!analytics) return null

  const totalDurMin = Math.round(analytics.total_duration_seconds / 60)

  return (
    <div className="min-h-screen bg-ivory text-gray-900 font-sans pb-12">
      <div className="max-w-6xl mx-auto p-6 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Behaviour Analytics</h1>
            <p className="text-sm text-gray-500 mt-1">Cross-session insights for your recording history</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={refresh}
              className="text-sm px-4 py-2 rounded-xl bg-white border border-gray-200 hover:bg-gray-50 text-gray-700 shadow-sm font-medium transition"
            >↻ Refresh</button>
            <button
              onClick={handleBulkExport}
              disabled={exporting || analytics.session_count === 0}
              className="text-sm px-4 py-2 rounded-xl bg-plum hover:bg-plum-dark text-white disabled:opacity-50 font-medium shadow-sm transition"
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
          <div className="lg:col-span-3 bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm rounded-2xl p-6">
            <h2 className="text-base font-semibold text-gray-900 mb-4">Metric Trends Across Sessions</h2>
            <MetricTrendChart trends={analytics.metric_trends} />
          </div>
          <div className="lg:col-span-2 bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm rounded-2xl p-6">
            <h2 className="text-base font-semibold text-gray-900 mb-4">Emotion Breakdown</h2>
            <EmotionChart breakdown={analytics.emotion_breakdown} />
          </div>
        </div>

        {/* Session table */}
        <div className="bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm rounded-2xl p-6">
          <h2 className="text-base font-semibold text-gray-900 mb-4">Session History</h2>
          <SessionTable sessions={analytics.sessions} />
        </div>
      </div>
    </div>
  )
}
