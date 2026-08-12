interface Props {
  label: string
  value: number      // [0, 1]
  color: string      // Tailwind stroke class like 'stroke-blue-500'
  size?: number
  waiting?: boolean
}

const COLOR_MAP: Record<string, string> = {
  'stroke-blue-400':   '#60a5fa',
  'stroke-green-400':  '#4ade80',
  'stroke-orange-400': '#fb923c',
  'stroke-red-400':    '#f87171',
}

const LEVEL_COLOR: Record<string, (v: number) => string> = {
  'stroke-red-400':    (v) => v > 0.7 ? '#ef4444' : v > 0.4 ? '#f87171' : '#fca5a5',
  'stroke-green-400':  (v) => v > 0.6 ? '#22c55e' : v > 0.3 ? '#4ade80' : '#86efac',
  'stroke-blue-400':   (v) => v > 0.6 ? '#3b82f6' : v > 0.3 ? '#60a5fa' : '#93c5fd',
  'stroke-orange-400': (v) => v > 0.65 ? '#f97316' : v > 0.35 ? '#fb923c' : '#fdba74',
}

export function MetricGauge({ label, value, color, size = 104, waiting = false }: Props) {
  const strokeW = 9
  const r = (size - strokeW * 2) / 2
  const cx = size / 2
  const cy = size / 2
  const circ = 2 * Math.PI * r
  const clamped = Math.max(0, Math.min(1, value))
  const offset = circ * (1 - clamped)

  // Adaptive colour based on value level
  const hex = waiting
    ? '#e5e7eb'
    : (LEVEL_COLOR[color]?.(clamped) ?? COLOR_MAP[color] ?? '#60a5fa')

  const pct = (clamped * 100).toFixed(0)

  return (
    <div className="flex flex-col items-center gap-1.5" role="meter" aria-valuenow={clamped * 100} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90" aria-hidden="true">
          {/* Track */}
          <circle
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke="#f3f4f6"
            strokeWidth={strokeW}
          />
          {/* Value arc */}
          <circle
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke={hex}
            strokeWidth={strokeW}
            strokeDasharray={circ}
            strokeDashoffset={waiting ? circ : offset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.5s cubic-bezier(0.4,0,0.2,1), stroke 0.4s ease' }}
          />
        </svg>
        {/* Centered label inside arc */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          {waiting ? (
            <div className="w-5 h-1.5 rounded-full bg-gray-200 animate-pulse" />
          ) : (
            <span className="text-base font-bold text-gray-900 leading-none tabular-nums">
              {pct}%
            </span>
          )}
        </div>
      </div>
      <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">{label}</span>
    </div>
  )
}
