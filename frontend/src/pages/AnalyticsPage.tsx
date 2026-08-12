import { useState } from 'react'
import {
  AreaChart, Area, BarChart, Bar,
  CartesianGrid, Legend, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { RefreshCw, Download, TrendingUp, Clock, Hash, Smile } from 'lucide-react'
import { exportSessionCsv, exportSessionJson, exportUserCsv } from '@/services/api'
import { useCurrentUser } from '@/hooks/useCurrentUser'
import { useUserAnalytics } from '@/hooks/useUserAnalytics'
import type { SessionSummary } from '@/types'

const METRIC_COLORS = {
  stress:     '#ef4444',
  engagement: '#22c55e',
  attention:  '#3b82f6',
  fatigue:    '#f97316',
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

function fmtShort(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function pct(v: number | null): string {
  return v != null ? `${Math.round(v * 100)}%` : '—'
}

// ── Summary Card ────────────────────────────────────────────────────────────

function SummaryCard({
  label, value, icon: Icon, accent = 'text-gray-900',
}: {
  label: string; value: string | number; icon: React.FC<{ className?: string }>; accent?: string
}) {
  return (
    <div className="bg-white border border-gray-100 shadow-sm rounded-2xl p-5 flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg bg-plum/8 flex items-center justify-center">
          <Icon className="h-4 w-4 text-plum" />
        </div>
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{label}</span>
      </div>
      <span className={`text-2xl font-bold ${accent}`}>{value}</span>
    </div>
  )
}

// ── Metric Trend Chart ──────────────────────────────────────────────────────

function MetricTrendChart({
  sessions,
  trends,
}: {
  sessions: SessionSummary[]
  trends: Record<string, { metric: string; points: { time: string; value: number }[] }>
}) {
  const sessionDates = sessions.map((s) => fmtShort(s.started_at))
  const labels = trends['stress']?.points.map((_, i) => sessionDates[i] ?? `S${i + 1}`) ?? []

  const data = labels.map((label, i) => ({
    label,
    stress:     trends['stress']?.points[i]?.value ?? 0,
    engagement: trends['engagement']?.points[i]?.value ?? 0,
    attention:  trends['attention']?.points[i]?.value ?? 0,
    fatigue:    trends['fatigue']?.points[i]?.value ?? 0,
  }))

  if (data.length === 0) {
    return (
      <div className="h-52 flex items-center justify-center text-gray-300 text-sm">
        No trend data yet
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={230}>
      <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
        <XAxis dataKey="label" tick={{ fill: '#9ca3af', fontSize: 11 }} tickLine={false} axisLine={{ stroke: '#e5e7eb' }} />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          tick={{ fill: '#9ca3af', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          width={44}
        />
        <Tooltip
          formatter={(v: number, name: string) => [`${Math.round(v * 100)}%`, name.charAt(0).toUpperCase() + name.slice(1)]}
          contentStyle={{ backgroundColor: '#fff', border: '1px solid #f3f4f6', borderRadius: '0.75rem', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.08)' }}
          labelStyle={{ color: '#374151', fontWeight: 600, fontSize: 11, marginBottom: 4 }}
          itemStyle={{ color: '#6b7280', fontSize: 12 }}
        />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} formatter={(v: string) => v.charAt(0).toUpperCase() + v.slice(1)} />
        {(Object.entries(METRIC_COLORS) as [string, string][]).map(([key, color]) => (
          <Area key={key} type="monotone" dataKey={key} stroke={color} fill={color} fillOpacity={0.08} strokeWidth={2} dot={{ r: 3, fill: color, strokeWidth: 0 }} activeDot={{ r: 5, strokeWidth: 0 }} />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ── Emotion Chart ───────────────────────────────────────────────────────────

function EmotionChart({ breakdown }: { breakdown: { counts: Record<string, number>; total: number } }) {
  const data = Object.entries(breakdown.counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), count }))

  if (data.length === 0) {
    return <div className="h-48 flex items-center justify-center text-gray-300 text-sm">No emotion data yet</div>
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" horizontal={false} />
        <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} tickLine={false} axisLine={{ stroke: '#e5e7eb' }} />
        <YAxis type="category" dataKey="name" width={80} tick={{ fill: '#6b7280', fontSize: 11 }} tickLine={false} axisLine={false} />
        <Tooltip
          formatter={(v: number) => [`${v} frames`, 'Count']}
          contentStyle={{ backgroundColor: '#fff', border: '1px solid #f3f4f6', borderRadius: '0.75rem', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.08)' }}
          itemStyle={{ color: '#6b7280', fontSize: 12 }}
        />
        <Bar dataKey="count" fill={EMOTION_COLOR} radius={[0, 6, 6, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Session Table ───────────────────────────────────────────────────────────

function SessionTable({ sessions }: { sessions: SessionSummary[] }) {
  const [exporting, setExporting] = useState<string | null>(null)

  async function doExport(sid: string, fmt: 'csv' | 'json') {
    setExporting(`${sid}-${fmt}`)
    try {
      if (fmt === 'csv') await exportSessionCsv(sid)
      else await exportSessionJson(sid)
    } finally { setExporting(null) }
  }

  if (sessions.length === 0) {
    return <p className="text-gray-400 text-sm text-center py-8">No sessions recorded yet.</p>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="text-xs text-gray-400 font-semibold uppercase tracking-wider border-b border-gray-100">
            <th className="py-3 pr-4">Date</th>
            <th className="py-3 pr-4">Context</th>
            <th className="py-3 pr-4">Duration</th>
            <th className="py-3 pr-4">Frames</th>
            <th className="py-3 pr-4">Stress</th>
            <th className="py-3 pr-4">Engagement</th>
            <th className="py-3 pr-4">Emotion</th>
            <th className="py-3">Export</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-50">
          {sessions.map((s) => (
            <tr key={s.session_id} className="hover:bg-gray-50/60 transition-colors">
              <td className="py-3 pr-4 text-gray-700 font-medium whitespace-nowrap">{fmtDate(s.started_at)}</td>
              <td className="py-3 pr-4 text-gray-500 max-w-[120px] truncate">{s.context ?? '—'}</td>
              <td className="py-3 pr-4 text-gray-400 tabular-nums">{fmtDur(s.duration_seconds)}</td>
              <td className="py-3 pr-4 text-gray-400 tabular-nums">{s.prediction_count}</td>
              <td className="py-3 pr-4">
                <span className={s.avg_stress != null && s.avg_stress > 0.65 ? 'text-red-600 font-semibold' : 'text-gray-600'}>
                  {pct(s.avg_stress)}
                </span>
              </td>
              <td className="py-3 pr-4">
                <span className={s.avg_engagement != null && s.avg_engagement > 0.6 ? 'text-green-600 font-semibold' : 'text-gray-600'}>
                  {pct(s.avg_engagement)}
                </span>
              </td>
              <td className="py-3 pr-4">
                <span className="capitalize text-plum font-semibold">{s.dominant_emotion ?? '—'}</span>
              </td>
              <td className="py-3">
                <div className="flex gap-1.5">
                  <button
                    onClick={() => doExport(s.session_id, 'csv')}
                    disabled={exporting === `${s.session_id}-csv`}
                    className="text-xs px-2.5 py-1 rounded-lg bg-white border border-gray-200 hover:bg-gray-50 text-gray-600 disabled:opacity-40 shadow-sm transition font-medium"
                  >CSV</button>
                  <button
                    onClick={() => doExport(s.session_id, 'json')}
                    disabled={exporting === `${s.session_id}-json`}
                    className="text-xs px-2.5 py-1 rounded-lg bg-white border border-gray-200 hover:bg-gray-50 text-gray-600 disabled:opacity-40 shadow-sm transition font-medium"
                  >JSON</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Main Page ───────────────────────────────────────────────────────────────

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
      <div className="min-h-screen bg-ivory flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-4 border-plum/20 border-t-plum rounded-full animate-spin" />
          <p className="text-sm text-gray-400 font-medium">Loading analytics…</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-ivory flex items-center justify-center">
        <div className="bg-white rounded-2xl border border-red-100 shadow-sm p-8 max-w-sm text-center">
          <div className="w-12 h-12 rounded-xl bg-red-50 flex items-center justify-center mx-auto mb-4">
            <TrendingUp className="h-6 w-6 text-red-400" />
          </div>
          <p className="text-sm text-gray-700 font-medium mb-4">{error}</p>
          <button
            onClick={refresh}
            className="px-5 py-2 rounded-xl bg-plum text-white text-sm font-semibold hover:bg-plum-dark shadow-sm transition"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!analytics) return null

  const totalDurMin = Math.round(analytics.total_duration_seconds / 60)
  const totalFrames = analytics.sessions.reduce((s, x) => s + x.prediction_count, 0)
  const dominantEmotion = Object.entries(analytics.emotion_breakdown.counts)
    .sort((a, b) => b[1] - a[1])[0]?.[0] ?? '—'

  return (
    <div className="min-h-screen bg-ivory text-gray-900 font-sans pb-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Behaviour Analytics</h1>
            <p className="text-sm text-gray-400 mt-1">Cross-session insights for your recording history</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={refresh}
              className="inline-flex items-center gap-1.5 text-sm px-4 py-2 rounded-xl bg-white border border-gray-200 hover:bg-gray-50 text-gray-600 shadow-sm font-medium transition"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </button>
            <button
              onClick={handleBulkExport}
              disabled={exporting || analytics.session_count === 0}
              className="inline-flex items-center gap-1.5 text-sm px-4 py-2 rounded-xl bg-plum hover:bg-plum-dark text-white disabled:opacity-50 font-semibold shadow-sm transition"
            >
              <Download className="h-3.5 w-3.5" />
              {exporting ? 'Exporting…' : 'Export All'}
            </button>
          </div>
        </div>

        {/* Summary cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <SummaryCard label="Sessions"    value={analytics.session_count} icon={Hash} />
          <SummaryCard label="Total Time"  value={`${totalDurMin} min`}    icon={Clock} />
          <SummaryCard label="Frames"      value={totalFrames}             icon={TrendingUp} />
          <SummaryCard
            label="Top Emotion"
            value={analytics.sessions.length > 0 ? dominantEmotion : '—'}
            icon={Smile}
            accent="text-plum capitalize"
          />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
          <div className="lg:col-span-3 bg-white border border-gray-100 shadow-sm rounded-2xl p-6">
            <h2 className="text-sm font-semibold text-gray-900 mb-1">Metric Trends Across Sessions</h2>
            <p className="text-xs text-gray-400 mb-4">Average values per session</p>
            <MetricTrendChart sessions={analytics.sessions} trends={analytics.metric_trends} />
          </div>
          <div className="lg:col-span-2 bg-white border border-gray-100 shadow-sm rounded-2xl p-6">
            <h2 className="text-sm font-semibold text-gray-900 mb-1">Emotion Breakdown</h2>
            <p className="text-xs text-gray-400 mb-4">Prediction counts by class</p>
            <EmotionChart breakdown={analytics.emotion_breakdown} />
          </div>
        </div>

        {/* Session table */}
        <div className="bg-white border border-gray-100 shadow-sm rounded-2xl p-6">
          <h2 className="text-sm font-semibold text-gray-900 mb-4">Session History</h2>
          <SessionTable sessions={analytics.sessions} />
        </div>
      </div>
    </div>
  )
}
