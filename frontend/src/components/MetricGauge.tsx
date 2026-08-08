interface Props {
  label: string
  value: number      // [0, 1]
  color: string      // Tailwind stroke class like 'stroke-blue-500'
  size?: number
}

function arc(value: number, r: number) {
  const circ = 2 * Math.PI * r
  return circ * (1 - value)   // strokeDashoffset
}

const COLOR_MAP: Record<string, string> = {
  'stroke-blue-400':   '#60a5fa',
  'stroke-green-400':  '#4ade80',
  'stroke-orange-400': '#fb923c',
  'stroke-red-400':    '#f87171',
}

export function MetricGauge({ label, value, color, size = 88 }: Props) {
  const r = (size - 12) / 2
  const circ = 2 * Math.PI * r
  const offset = arc(Math.max(0, Math.min(1, value)), r)
  const hex = COLOR_MAP[color] ?? '#60a5fa'
  const pct = Math.round(value * 100)

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke="#f3f4f6" strokeWidth={8} />
        <circle cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke={hex} strokeWidth={8}
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.4s ease' }}
        />
      </svg>
      <span className="text-lg font-bold text-gray-900 -mt-1">{pct}%</span>
      <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
    </div>
  )
}
