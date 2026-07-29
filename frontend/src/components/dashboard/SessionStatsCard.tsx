/**
 * SessionStatsCard — compact summary of session aggregate metrics.
 * Shown below the session list when a completed session is selected.
 */
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

const METRIC_ROWS: Array<{ label: string; key: keyof SessionStats; fmt: (v: SessionStats[keyof SessionStats]) => string }> = [
  { label: 'Avg Stress',      key: 'avg_stress',      fmt: (v) => pct(v as number | null) },
  { label: 'Avg Engagement',  key: 'avg_engagement',  fmt: (v) => pct(v as number | null) },
  { label: 'Avg Attention',   key: 'avg_attention',   fmt: (v) => pct(v as number | null) },
  { label: 'Avg Fatigue',     key: 'avg_fatigue',     fmt: (v) => pct(v as number | null) },
]

export function SessionStatsCard({ stats, loading }: Props) {
  return (
    <div className={clsx('mt-4 rounded-lg border border-gray-700 p-4 text-xs', loading && 'opacity-50')}>
      <p className="mb-3 font-semibold text-gray-300">Session Summary</p>

      <div className="mb-3 grid grid-cols-2 gap-2">
        <div className="rounded-md bg-gray-900 p-2">
          <p className="text-gray-500">Predictions</p>
          <p className="mt-0.5 font-mono text-white">{stats.prediction_count}</p>
        </div>
        <div className="rounded-md bg-gray-900 p-2">
          <p className="text-gray-500">Duration</p>
          <p className="mt-0.5 font-mono text-white">{dur(stats.duration_seconds)}</p>
        </div>
        <div className="col-span-2 rounded-md bg-gray-900 p-2">
          <p className="text-gray-500">Dominant Emotion</p>
          <p className="mt-0.5 font-mono capitalize text-white">{stats.dominant_emotion ?? '—'}</p>
        </div>
      </div>

      <div className="space-y-1.5">
        {METRIC_ROWS.map(({ label, key, fmt }) => (
          <div key={key} className="flex items-center justify-between">
            <span className="text-gray-500">{label}</span>
            <span className="font-mono text-white">{fmt(stats[key])}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
