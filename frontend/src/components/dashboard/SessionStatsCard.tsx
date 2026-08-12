import { clsx } from 'clsx'
import type { SessionStats } from '@/types'

interface Props {
  stats: SessionStats
  loading: boolean
}

function pct(v: number | null): string {
  return v === null ? '—' : `${Math.round(v * 100)}%`
}

function dur(seconds: number | null): string {
  if (seconds === null) return '—'
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

interface MetricRowProps {
  label: string
  value: string
  accent?: string
}

function MetricRow({ label, value, accent }: MetricRowProps) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-50 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={clsx('text-xs font-semibold tabular-nums', accent ?? 'text-gray-800')}>{value}</span>
    </div>
  )
}

export function SessionStatsCard({ stats, loading }: Props) {
  const stressHigh  = stats.avg_stress     != null && stats.avg_stress > 0.65
  const fatigueHigh = stats.avg_fatigue    != null && stats.avg_fatigue > 0.6
  const engHigh     = stats.avg_engagement != null && stats.avg_engagement > 0.6

  return (
    <div className={clsx(
      'mt-4 rounded-xl border border-gray-100 bg-gray-50/60 p-4 transition-opacity',
      loading && 'opacity-60',
    )}>
      <p className="text-xs font-bold text-gray-700 uppercase tracking-wider mb-3">Session Summary</p>

      {/* Key stats grid */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="rounded-lg bg-white border border-gray-100 shadow-sm p-2.5 text-center">
          <p className="text-[10px] text-gray-400 mb-0.5">Predictions</p>
          <p className="text-base font-bold text-gray-900 tabular-nums">{stats.prediction_count}</p>
        </div>
        <div className="rounded-lg bg-white border border-gray-100 shadow-sm p-2.5 text-center">
          <p className="text-[10px] text-gray-400 mb-0.5">Duration</p>
          <p className="text-base font-bold text-gray-900">{dur(stats.duration_seconds)}</p>
        </div>
        <div className="rounded-lg bg-white border border-gray-100 shadow-sm p-2.5 text-center">
          <p className="text-[10px] text-gray-400 mb-0.5">Emotion</p>
          <p className="text-xs font-bold text-plum capitalize truncate">{stats.dominant_emotion ?? '—'}</p>
        </div>
      </div>

      {/* Metric rows */}
      <div>
        <MetricRow label="Avg Stress"      value={pct(stats.avg_stress)}      accent={stressHigh  ? 'text-red-600'   : undefined} />
        <MetricRow label="Avg Engagement"  value={pct(stats.avg_engagement)}  accent={engHigh     ? 'text-green-600' : undefined} />
        <MetricRow label="Avg Attention"   value={pct(stats.avg_attention)} />
        <MetricRow label="Avg Fatigue"     value={pct(stats.avg_fatigue)}     accent={fatigueHigh ? 'text-orange-600': undefined} />
      </div>
    </div>
  )
}
